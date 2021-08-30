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
from dataset import NL2SPARQLDataset
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

# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )
#
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s
#
# def read_txt_file(file, reverse=False):
#     # Read the file and split into lines
#     lines = open(file, encoding='utf-8').read().strip().split('\n')
#
#     # Split every line into pairs and normalize
#     pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
#
#     return pairs
#
# def read_csv_file(file, reverse=False):
#     # Read csv file
#     with open(file, 'r') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         pairs = []
#         for row in csv_reader:
#             pairs.append([row[0], row[1]])
#
#         # pairs = [[normalizeString(s) for s in row.split(',')[:2]] for row in csv_reader]
#
#     return pairs

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

    fh = logging.FileHandler(os.path.join(save_dir, args.expname_posgt2uri) + '.log', mode='w')
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenize = lambda x: x.split()

    text_field = Field(tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
    uri_field = Field(tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>")

    fields = {'text': ('src', text_field), 'uri': ('trg', uri_field)}

    file_train_dir = 'train/posgtu.csv'
    file_valid_dir = 'valid/posgtu.csv'
    file_test_dir = 'test/posgtu.csv'

    train_data, valid_data, test_data = TabularDataset.splits(path=args.data,
                                                              train=file_train_dir,
                                                              test=file_valid_dir,
                                                              validation=file_test_dir,
                                                              format="csv",
                                                              fields=fields)

    text_field.build_vocab(train_data, max_size=100000, min_freq=1)
    uri_field.build_vocab(train_data, max_size=100000, min_freq=1)

    print("%%%" * 30)
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    # Training hyperparameters
    # num_epochs = 10000
    learning_rate = 3e-4
    batch_size = 32
    train_batch_size = 32
    valid_batch_size = 32
    test_batch_size = 64
    # Model hyperparameters

    src_vocab_size = len(text_field.vocab)
    trg_vocab_size = len(uri_field.vocab)
    print(f"src_vocab_size: {src_vocab_size} , trg_vocab_size: {trg_vocab_size}")

    embedding_size = 300
    num_heads = 10
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 500
    forward_expansion = 4
    # q_pad_idx = vocab_q.getIndex("<pad>")
    src_pad_idx = text_field.vocab.stoi["<pad>"]
    last_check = 0

    # dataloader = {
    #     'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    #     'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    # }
    #
    # print(f'Created `torch_dataloader train` with {len(dataloader["train"])} batches!')
    # print(f'Created `torch_dataloader test` with {len(dataloader["test"])} batches!')

    # dataloader['train'].create_batches()
    # batch_temp = next(iter(dataloader['train'].batches))
    # print(batch_temp)
    # # # # print(batch_temp[:]['query'], batch_temp[:]['sparql'])

    # # Group similar length text sequences together in batches.
    # train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    #     # Datasets for iterator to draw data from
    #     (train_dataset, valid_dataset, test_dataset),
    #     # Tuple of train and validation batch sizes.
    #     batch_sizes=(train_batch_size, valid_batch_size, test_batch_size),
    #     # Device to load batches on.
    #     device=device,
    #     # Function to use for sorting examples.
    #     sort_key=lambda x: len(x['question']),
    #     # Repeat the iterator for multiple epochs.
    #     repeat=True,
    #     # Sort all examples in data using `sort_key`.
    #     sort=False,
    #     # Shuffle data on each epoch run.
    #     shuffle=True,
    #     # Use `sort_key` to sort examples in each batch.
    #     sort_within_batch=True,
    # )

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        # batch_size=(train_batch_size, valid_batch_size, test_batch_size),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    train_batch = next(iter(train_iterator))
    print(train_batch)

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
    pad_idx = uri_field.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training

    num_epochs = 10

    checkpoint_filename = '%s.pt' % os.path.join(args.save, args.expname_posgt2uri)
    if args.mode == "test":
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['model'])
        args.epochs = 1

    # create trainer object for training and testing
    trainer = Trainer(args, model, device, criterion, optimizer, args.epochs)

    for epoch in range(args.epochs):
        if args.mode == "train":
            # scheduler.step()

            train_loss = trainer.train(train_iterator)
            test_loss, train_pred = trainer.test(test_iterator, max_len)
            logger.info(
                '==> Epoch {}, Train \tLoss: {} -- {}'.format(epoch, train_loss, test_loss))
            # logger.info(
            #     '==> Epoch {}, Train \tLoss: {} {}'.format(epoch, train_loss,
            #                                                metrics.all(train_pred, train_dataset.labels)))

            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'args': args, 'epoch': epoch}
            # checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
            #               'args': args, 'epoch': epoch, 'scheduler': scheduler}

            checkpoint_filename = '%s.pt' % os.path.join(args.save,
                                                         args.expname_posgt2uri + ',epoch={},train_loss={}'.format(epoch + 1,
                                                                                                       train_loss))
            torch.save(checkpoint, checkpoint_filename)


        # dev_loss, dev_pred = trainer.test(dev_dataset)
        # test_loss, test_pred = trainer.test(test_dataset)
        # logger.info(
        #     '==> Epoch {}, Dev \tLoss: {} {}'.format(epoch, dev_loss, metrics.all(dev_pred, dev_dataset.labels)))
        # logger.info(
        #     '==> Epoch {}, Test \tLoss: {} {}'.format(epoch, test_loss, metrics.all(test_pred, test_dataset.labels)))


if __name__ == "__main__":
    main()
