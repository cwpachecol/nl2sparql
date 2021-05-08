"""
Seq2Seq using Transformers on the Multi30k
dataset. In this video I utilize Pytorch
inbuilt Transformer modules, and have a
separate implementation for Transformers
from scratch. Training this model for a
while (not too long) gives a BLEU score
of ~35, and I think training for longer
would give even better results.

"""
import sys
import os
import io

import torch
import torch.nn as nn
import torch.optim as optim
# import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
# from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

# from google.colab import drive
# drive.mount('/content/drive')

# import DataSplit as ds
# import generator_utils as gu
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

path_datasets = 'data/DBNQA'

tokenize = lambda x: x.split()

question_field = Field(tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
sparql_field = Field(tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>")

fields = {'question': ('src', question_field), 'sparql': ('trg', sparql_field)}

train_data, valid_data, test_data = TabularDataset.splits(path=path_datasets,
                                                               train="DBNQA_train.csv",
                                                               test="DBNQA_test.csv",
                                                               validation="DBNQA_valid.csv",
                                                               format="csv",
                                                               fields=fields)

print(len(train_data))
print(vars(train_data[0]))
print(len(test_data))
print(vars(test_data[0]))
print(len(valid_data))
print(vars(valid_data[0]))

question_field.build_vocab(train_data, max_size=100000, min_freq=1)
sparql_field.build_vocab(train_data, max_size=100000, min_freq=1)

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_mode = False
valid_mode = False
test_mode = True
load_model = True
save_model = False

# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4
batch_size = 120
# Model hyperparameters

src_vocab_size = len(question_field.vocab)
trg_vocab_size = len(sparql_field.vocab)
print(f"src_vocab_size: { src_vocab_size } , trg_vocab_size: { trg_vocab_size }")

embedding_size = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 1010
forward_expansion = 4
src_pad_idx = sparql_field.vocab.stoi["<pad>"]
last_check = 334

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = sparql_field.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    checks_path = str('checkpoints/DBNQA/cp_DBNQA_') + str(last_check) + str('_epochs.pth.tar')
    load_checkpoint(checks_path, model, optimizer, device)

if train_mode:
    sentence = "Is Alexander Hamilton a lawyer?"

    for epoch in range(last_check + 1, num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        if save_model:
            # if (epoch + last_check) % 5 == 0:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename='checkpoints/DBNQA/cp_DBNQA_' + str(epoch) + '_epochs.pth.tar')

        model.eval()
        translated_sentence = translate_sentence(model, sentence, question_field, sparql_field, device, max_length=50)

        print(f"Sentence: {sentence} \n")
        print(f"Translated example sentence: {translated_sentence} \n ")
        # print(" ".join(translated_sentence))

        model.train()
        losses = []

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        print(f"loss mean: {(mean_loss)}")
        scheduler.step(mean_loss)

if test_mode:
    targets = []
    outputs = []

    s_dbpedia_json = open(os.path.join(path_datasets, 'sparql_dbpedia.json'), 'w', encoding="utf-8")
    s_dbpedia_txt = open(os.path.join(path_datasets, 'sparql_dbpedia.txt'), 'w', encoding="utf-8")

    for example in test_data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, question_field, sparql_field, device, max_length=500)
        prediction = prediction[:-1]  # remove <eos> token

        # sparql_wikidata = q2sparql[line]
        # sparql_wikidata = str(re.split('\)\n(?=ns)|[ ]*\.\n[\t]*|\{\n[\t]*(?=ns)', q2sparql[line]))
        sparql_dbpedia = str(prediction).strip().lower().replace('\n', ' ').replace('\t', '').replace('\r', ' ')
        sparql_dbpedia = sparql_dbpedia.replace("     ", " ").replace("    ", " ").replace("   ", " ").replace("  ",
                                                                                                               " ")
        # print(sparql_wikidata)
        s_dbpedia_txt.write(sparql_dbpedia)
        s_dbpedia_txt.write('\n')

        targets.append([trg])
        outputs.append(prediction)

    s_dbpedia_json.close()
    s_dbpedia_txt.close()

if valid_mode:
    # running on entire test data takes a while
    score = bleu(test_data[1:100], model, question_field, sparql_field, device)
    print(f"Bleu score {score * 100:.2f}")