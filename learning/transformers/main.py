from __future__ import division
from __future__ import print_function

import os
import csv
import random
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import sys
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

# IMPORT CONSTANTS
import constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import QGDataset, NL2SPARQLDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from learning.transformers.utils import load_word_vectors, build_vocab
# CONFIG PARSER
from learning.transformers.config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer
import datetime
from fasttext import load_model

# sys.path.insert(0, os.path.abspath("..//.."))
sys.path.insert(0, os.path.abspath('../..'))

import unicodedata
import re

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_txt_file(file, reverse=False):
    # Read the file and split into lines
    lines = open(file, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    return pairs

def read_csv_file(file, reverse=False):
    # Read csv file
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        pairs = []
        for row in csv_reader:
            pairs.append([row[0], row[1]])

        # pairs = [[normalizeString(s) for s in row.split(',')[:2]] for row in csv_reader]

    return pairs

def main():
    global args
    args = parse_args()
    # global logger

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    # base_dir = os.path.dirname(os.path.realpath( __file__ ))
    base_dir = sys.path[0]
    # file logger
    # arg_save = "Tree-LSTM"
    # arg_expname = "Tree-LSTM"
    # print("-"*50)
    # print(base_dir)
    # print(args.save)
    save_dir = base_dir + '/' + args.save
    # print(save_dir)

    fh = logging.FileHandler(os.path.join(save_dir, args.expname) + '.log', mode='w')
    # fh = logging.FileHandler(os.path.join(arg_save, arg_expname) + '.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')

    logger.debug(args)
    # args.data = 'learning/treelstm/data/LC-QUAD10/'
    # args.save = 'learning/treelstm/checkpoints/'

    args.data = 'data/lcquad10/'
    args.save = 'checkpoints/'

    torch.manual_seed(args.seed)
    # random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    valid_dir = os.path.join(args.data, 'valid/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files for questions
    dataset_vocab_file_q = os.path.join(args.data, 'dataset_q.vocab')
    if not os.path.isfile(dataset_vocab_file_q):
        token_files_q = [os.path.join(split, 'q.txt') for split in [train_dir, valid_dir, test_dir]]
        # token_files_s = [os.path.join(split, 's.txt') for split in [train_dir, valid_dir, test_dir]]
        # token_files = token_files_q + token_files_s
        dataset_vocab_file_q = os.path.join(args.data, 'dataset_q.vocab')
        build_vocab(token_files_q, dataset_vocab_file_q)

    # write unique words from all token files for sparqls
    dataset_vocab_file_s = os.path.join(args.data, 'dataset_s.vocab')
    if not os.path.isfile(dataset_vocab_file_s):
        token_files_s = [os.path.join(split, 's.txt') for split in [train_dir, valid_dir, test_dir]]
        # token_files_s = [os.path.join(split, 's.txt') for split in [train_dir, valid_dir, test_dir]]
        # token_files = token_files_q + token_files_s
        dataset_vocab_file_s = os.path.join(args.data, 'dataset_s.vocab')
        build_vocab(token_files_s, dataset_vocab_file_s)

    # get vocab questions object from vocab file previously written
    vocab_q = Vocab(filename=dataset_vocab_file_q,
                  data=[constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD])
    logger.debug('==> Dataset vocabulary questions size : %d ' % vocab_q.size())

    # get vocab sparqls object from vocab file previously written
    vocab_s = Vocab(filename=dataset_vocab_file_s,
                  data=[constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD])
    logger.debug('==> Dataset vocabulary questions size : %d ' % vocab_s.size())

    # load dataset splits
    MAX_LENGTH = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_file = os.path.join(args.data, 'dataset_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = NL2SPARQLDataset(train_dir, vocab_q, vocab_s, MAX_LENGTH, device=device)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    valid_file = os.path.join(args.data, 'dataset_valid.pth')
    if os.path.isfile(valid_file):
        valid_dataset = torch.load(valid_file)
    else:
        valid_dataset = NL2SPARQLDataset(valid_dir, vocab_q, vocab_s, MAX_LENGTH, device=device)
        torch.save(valid_dataset, valid_file)
    logger.debug('==> Size of valid data     : %d ' % len(valid_dataset))
    test_file = os.path.join(args.data, 'dataset_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = NL2SPARQLDataset(test_dir, vocab_q, vocab_s, MAX_LENGTH, device=device)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    query_sentence = train_dataset[0][0]
    sparql_sentence = train_dataset[0][1]

    print(vocab_q.convertToLabels(query_sentence.tolist(), 0))
    print(vocab_s.convertToLabels(sparql_sentence.tolist(), 0))

    # print(train_dataset[0][0])
    # print("==++"*20)
    # print(train_dataset[:5][0])
    # print("&&&&" * 20)
    # for e in train_dataset[:5][0]:
    #     # print(e.squeeze().tolist())
    #     # print(e.flatten().tolist())
    #     print(e.tolist())
    #
    #     # print(vocab.convertToLabels(e.flatten().tolist(), 1880))
    #     print(vocab.convertToLabels(e.tolist(), 3))

    # Training hyperparameters
    num_epochs = 10000
    learning_rate = 3e-4
    batch_size = 32
    # Model hyperparameters

    src_vocab_size = vocab_q.size()
    trg_vocab_size = vocab_s.size()
    print(f"src_vocab_size: {src_vocab_size} , trg_vocab_size: {trg_vocab_size}")

    embedding_size = 256
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 1000
    forward_expansion = 4
    q_pad_idx = vocab_q.getIndex("<pad>")
    last_check = 0

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x[0]),
        device=device,
    )

    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        q_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=0.1, patience=10, verbose=True
# )
#
    s_pad_idx = vocab_s.getIndex("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=s_pad_idx)
#
# if load_model:
#     checks_path = str('/content/drive/MyDrive/Colab Notebooks/nl_to_sparql/checks/lcquad10/check_nl_to_sparql_') + str(
#         last_check) + str('_epochs.pth.tar')
#     print(checks_path)
#
#     if device == 'cpu':
#         load_checkpoint(torch.load(checks_path, map_location=torch.device('cpu')), model, optimizer)
#     else:
#         load_checkpoint(checks_path, model, optimizer)
#         # load_checkpoint(torch.load('/content/drive/MyDrive/Colab Notebooks/nl_to_sparql/checks/lcquad10/check_nl_to_sparql_10_epochs.pth.tar'), model, optimizer)
#         # with open(checks_path, 'rb') as f:
#         #   buffer = io.BytesIO(f.read())
#         #   load_checkpoint(torch.load(buffer), model, optimizer)

    exit()
# Training
    sentence = "Which city's foundeer is John Forbes?"

    for epoch in range(last_check + 1, num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        # if save_model:
        #     # if (epoch + last_check) % 5 == 0:
        #     checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        #     save_checkpoint(checkpoint,
        #                     filename='/content/drive/MyDrive/Colab Notebooks/nl_to_sparql/checks/lcquad10/check_nl_to_sparql_' + str(
        #                         epoch) + '_epochs.pth.tar')
        #
        # model.eval()
        # translated_sentence = translate_sentence(model, sentence, question_field, sparql_field, device, max_length=50)
        #
        # print(f"Sentence: {sentence} \n")
        # print(f"Translated example sentence: {translated_sentence} \n ")
        # # print(" ".join(translated_sentence))

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
            # writer.add_scalar("Training loss", loss, global_step=step)
            # step += 1

        mean_loss = sum(losses) / len(losses)
        print(f"loss mean: {(mean_loss)}")
        # scheduler.step(mean_loss)

    exit()

    criterion = nn.KLDivLoss()  # nn.HingeEmbeddingLoss()

    if args.cuda:
        model.cuda(), criterion.cuda()
    else:
        torch.set_num_threads(4)
    logger.info("number of available cores: {}".format(torch.get_num_threads()))
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'dataset_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        EMBEDDING_DIM = 300
        emb = torch.zeros(vocab.size(), EMBEDDING_DIM, dtype=torch.float)
        fasttext_model = load_model("data/fasttext/wiki.en.bin")
        print('Use Fasttext Embedding')
        for word in vocab.labelToIdx.keys():
            word_vector = fasttext_model.get_word_vector(word)
            if word_vector.all() != None and len(word_vector) == EMBEDDING_DIM:
                emb[vocab.getIndex(word)] = torch.Tensor(word_vector)
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(EMBEDDING_DIM).uniform_(-1, 1)
        # # load glove embeddings and vocab
        # args.glove = 'learning/treelstm/data/glove/'
        # print('Use Glove Embedding')
        # glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
        # logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        # emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(-0.05, 0.05)
        # # zero out the embeddings for padding and other special words if they are absent in vocab
        # for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
        #     emb[idx].zero_()
        # for word in vocab.labelToIdx.keys():
        #     if glove_vocab.getIndex(word):
        #         emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    model.emb.weight.data.copy_(emb)

    checkpoint_filename = '%s.pt' % os.path.join(args.save, args.expname)
    if args.mode == "test":
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['model'])
        args.epochs = 1

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)

    for epoch in range(args.epochs):
        if args.mode == "train":
            scheduler.step()

            train_loss = trainer.train(train_dataset)
            train_loss, train_pred = trainer.test(train_dataset)
            logger.info(
                '==> Epoch {}, Train \tLoss: {} {}'.format(epoch, train_loss,
                                                           metrics.all(train_pred, train_dataset.labels)))
            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'args': args, 'epoch': epoch, 'scheduler': scheduler}
            checkpoint_filename = '%s.pt' % os.path.join(args.save,
                                                         args.expname + ',epoch={},train_loss={}'.format(epoch + 1,
                                                                                                       train_loss))
            torch.save(checkpoint, checkpoint_filename)

        dev_loss, dev_pred = trainer.test(dev_dataset)
        test_loss, test_pred = trainer.test(test_dataset)
        logger.info(
            '==> Epoch {}, Dev \tLoss: {} {}'.format(epoch, dev_loss, metrics.all(dev_pred, dev_dataset.labels)))
        logger.info(
            '==> Epoch {}, Test \tLoss: {} {}'.format(epoch, test_loss, metrics.all(test_pred, test_dataset.labels)))


if __name__ == "__main__":
    main()
