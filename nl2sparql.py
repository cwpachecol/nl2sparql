import sys
import os
import io
import json
import numpy as np
from pathlib import Path

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
import pickle
from generator_utils import decode, fix_URI
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

tokenize = lambda x: x.split()

def load_dataset(path_datasets, train_file, test_file, valid_file, batch_size=1, device='cpu'):
    question_field = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
    sparql_field = Field(tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
    fields = {'question': ('src', question_field), 'sparql': ('trg', sparql_field)}
    train_data, valid_data, test_data = TabularDataset.splits(path=path_datasets,
                                                              train=train_file,
                                                              test=test_file,
                                                              validation=valid_file,
                                                              format="csv",
                                                              fields=fields,
                                                              skip_header=False)
    # question_vocab = question_field.build_vocab(train_data, max_size=None, min_freq=1)
    question_vocab = question_field.build_vocab(train_data, max_size=1000, vectors="glove.6B.100d")
    sparql_vocab = sparql_field.build_vocab(train_data, max_size=None, min_freq=1)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )


    return train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data, question_field, sparql_field

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


def fit(train_iterator, model, device, load_model, checkpoint_path, last_checkpoint, epochs, learning_rate, pad_idx):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint_path = ""
        load_checkpoint_path = str(checkpoint_path) + str('/cp_') + str(dataset_name) + str('_') + str(last_checkpoint) + str('_epochs.pth.tar')

        if Path(load_checkpoint_path).is_file():
            load_checkpoint(load_checkpoint_path, model, optimizer, device)
            print(f"model: {load_checkpoint_path} loaded..")
        else:
            print(f"Checkpoint: {load_checkpoint_path} fiel, no find, start from scratch")

    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

    # Tensorboard to get nice loss plot
    writer = SummaryWriter("runs/loss_plot")
    step = 0

    sentence = "Is Alexander Hamilton a lawyer?"
    bar = tqdm(range(last_checkpoint + 1, epochs + last_checkpoint + 1))
    for epoch in bar:
        print(f"[Epoch {epoch} / {epochs + last_checkpoint}]")
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
        bar.set_description(f"loss {np.mean(losses):.5f}")

        scheduler.step(mean_loss)

        if save_model:
            # if (epoch + last_check) % 5 == 0:

            save_checkpoint_path = ""
            save_checkpoint_path = str(checkpoint_path) + str('/cp_') + str(dataset_name) + str('_') + str(
                epoch) + str('_epochs.pth.tar')
            print(f"Saving checkpoint:{ save_checkpoint_path }")
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, save_checkpoint_path)

        model.eval()
        translated_sentence = translate_sentence(model, sentence, question_field, sparql_field, device, max_length=50)
        print(f"Sentence: {sentence} \n")
        print(f"Translated example sentence: {translated_sentence} \n ")

def test(test_data, question_field, sparql_field, model, device, path_checkpoint, learning_rate, pad_idx):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if Path(path_checkpoint).is_file():
            load_checkpoint(path_checkpoint, model, optimizer, device)
            print(f"model: {path_checkpoint} loaded..")

        model.to(device)

        targets = []
        outputs = []

        s_dbpedia_json = open(os.path.join(path_datasets, 'sparql_dbpedia.json'), 'w', encoding="utf-8")
        s_dbpedia_txt = open(os.path.join(path_datasets, 'sparql_dbpedia.txt'), 'w', encoding="utf-8")

        cont = 1
        for example in test_data:
            src = vars(example)["src"]
            trg = vars(example)["trg"]

            prediction = translate_sentence(model, src, question_field, sparql_field, device, max_length=500)
            prediction = prediction[:-1]  # remove <eos> token
            print(f"row: {cont} of {len(test_data)}")
            print(f"Question: {src} \nSparql grand true: {trg} \nSparql predicted: {prediction}")
            # sparql_wikidata = q2sparql[line]
            # sparql_wikidata = str(re.split('\)\n(?=ns)|[ ]*\.\n[\t]*|\{\n[\t]*(?=ns)', q2sparql[line]))
            sparql_dbpedia = str(prediction).strip().lower().replace('\n', ' ').replace('\t', '').replace('\r', ' ')
            sparql_dbpedia = sparql_dbpedia.replace("     ", " ").replace("    ", " ").replace("   ", " ").replace("  ",
                                                                                                                   " ")
            # print(sparql_wikidata)
            s_dbpedia_txt.write(str(src) + ", " + str(sparql_dbpedia))
            s_dbpedia_txt.write('\n')

            targets.append([trg])
            outputs.append(prediction)
            cont = cont + 1

        s_dbpedia_json.close()
        s_dbpedia_txt.close()

def predict(question, question_field, sparql_field, model, device, path_checkpoint, learning_rate, pad_idx):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if Path(path_checkpoint).is_file():
        load_checkpoint(path_checkpoint, model, optimizer, device)
        print(f"model: {path_checkpoint} loaded..")

    model.to(device)
    prediction = translate_sentence(model, question, question_field, sparql_field, device, max_length=1000)
    prediction = prediction[:-1]  # remove <eos> token
    return prediction

def valid(valid_data, question_field, sparql_field, model, device, path_checkpoint, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if Path(path_checkpoint).is_file():
        load_checkpoint(path_checkpoint, model, optimizer, device)
        print(f"model: {path_checkpoint} loaded..")

    model.to(device)

    score = bleu(valid_data, model, question_field, sparql_field, device)
    print(f"Bleu score {score * 100:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", dest="config_file", help="name of the config file (*.json).")

    args = parser.parse_args()

    with open(args.config_file, "r") as read_file:
        args_from_file = json.load(read_file)

    # Prepare path's and files
    dataset_name = args_from_file['dataset_name']
    dataset_path = args_from_file['dataset_path'] + '/' + dataset_name
    checkpoint_path = args_from_file['checkpoint_path'] + '/' + dataset_name

    train_file = dataset_name + '_train.csv'
    test_file = dataset_name + '_test.csv'
    valid_file = dataset_name + '_valid.csv'
    batch_size = args_from_file['batch_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing in: {device}")

    train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data, question_field, sparql_field = load_dataset(dataset_path, train_file, test_file, valid_file, batch_size, device)
    src_vocab_size = len(question_field.vocab)
    trg_vocab_size = len(sparql_field.vocab)
    print(f"Vocabulary:\nsrc_vocab_size: {src_vocab_size} , trg_vocab_size: {trg_vocab_size}")

    # We're ready to define everything we need for training our Seq2Seq model
    train_mode = args_from_file['train_mode']
    valid_mode = args_from_file['test_mode']
    test_mode = args_from_file['valid_mode']
    predict_mode = args_from_file['predict_mode']
    load_model = args_from_file['load_mode']
    save_model = args_from_file['save_mode']

    # Training hyperparameters
    epochs = args_from_file['epochs']
    learning_rate = args_from_file['learning_rate']

    # Model hyperparameters
    embedding_size = 256
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 1010
    forward_expansion = 4
    src_pad_idx = question_field.vocab.stoi["<pad>"]
    last_checkpoint = args_from_file['last_checkpoint']

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

    # checkpoint_path = ""
    # if load_model:
    #     checkpoint_path = str(checkpoint_path) + str('/cp_') + str(dataset_name) + str('_') + str(
    #         last_checkpoint) + str('_epochs.pth.tar')

    if train_mode:
        fit(train_iterator, model, device, load_model, checkpoint_path, last_checkpoint, epochs, learning_rate, src_pad_idx)

    if test_mode:
        test(test_data, question_field, sparql_field, model, device, checkpoint_path, learning_rate, src_pad_idx)

    if predict_mode:
        question_nl = ""
        while(question_nl != '-q'):
            question_nl = input("Input natural language question or write '-q' to finish:")
            sparql_predicted_encoded = predict(question_nl, question_field, sparql_field, model, device, checkpoint_path, learning_rate, src_pad_idx)
            # print(f"predicted sparql encoded: { sparql_predicted_encoded }")
            sparql_predicted_decoded = "".join(sparql_predicted_encoded)
            # print(sparql_predicted_decoded)
            sparql_predicted_decoded = decode(sparql_predicted_decoded)
            # print(sparql_predicted_decoded)
            # sparql_predicted_decoded = fix_URI(sparql_predicted_decoded)
            print(f"sparql predicted: { sparql_predicted_decoded }")

    if valid_mode:
        valid(valid_data, question_field, sparql_field, model, device, checkpoint_path, learning_rate)