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
# IMPORT CONSTANTS
import constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import QGDataset, NL2SPARQLDataset, Lang
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

    # with open(file) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     pairs = []
    #     for row in csv_reader:
    #         pairs.append(row)
    #     # line_count = 0
        # for row in csv_reader:
        #     if line_count == 0:
        #         print(f'Column names are {", ".join(row)}')
        #         line_count += 1
        #     else:
        #         print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
        #         line_count += 1
        # print(f'Processed {line_count} lines.')

    # return pairs


def prepareData(file, filters=None, max_length=None, reverse=False):
    pairs = read_csv_file(file, reverse)
    print(f"Tenemos {len(pairs)} pares de frases")

    # if filters is not None:
    #     assert max_length is not None
    #     pairs = filterPairs(pairs, filters, max_length, int(reverse))
    #     print(f"Filtramos a {len(pairs)} pares de frases")

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang('sparql')
        output_lang = Lang('query')
    else:
        input_lang = Lang('query')
        output_lang = Lang('sparql')

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

        # add <eos> token
        pair[0] += " EOS"
        pair[1] += " EOS"

    print("Longitud vocabularios:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs



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

    # write unique words from all token files
    dataset_vocab_file = os.path.join(args.data, 'dataset.vocab')
    if not os.path.isfile(dataset_vocab_file):
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, valid_dir, test_dir]]
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, valid_dir, test_dir]]
        token_files = token_files_a + token_files_b
        dataset_vocab_file = os.path.join(args.data, 'dataset.vocab')
        build_vocab(token_files, dataset_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=dataset_vocab_file,
                  data=[constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD])
    logger.debug('==> Dataset vocabulary size : %d ' % vocab.size())

    # load dataset splits
    # train_file = os.path.join(args.data, 'dataset_train.pth')
    # if os.path.isfile(train_file):
    #     train_dataset = torch.load(train_file)
    # else:
    #     train_dataset = QGDataset(train_dir, vocab, args.num_classes)
    #     torch.save(train_dataset, train_file)
    # logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    # valid_file = os.path.join(args.data, 'dataset_valid.pth')
    # if os.path.isfile(valid_file):
    #     valid_dataset = torch.load(valid_file)
    # else:
    #     valid_dataset = QGDataset(valid_dir, vocab, args.num_classes)
    #     torch.save(valid_dataset, valid_file)
    # logger.debug('==> Size of valid data     : %d ' % len(valid_dataset))
    # test_file = os.path.join(args.data, 'dataset_test.pth')
    # if os.path.isfile(test_file):
    #     test_dataset = torch.load(test_file)
    # else:
    #     test_dataset = QGDataset(test_dir, vocab, args.num_classes)
    #     torch.save(test_dataset, test_file)
    # logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    # load dataset splits
    MAX_LENGTH = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_file = os.path.join(args.data, 'dataset_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        raw_train_file = os.path.join(args.data, 'train/qs.csv')
        input_lang, output_lang, pairs = prepareData(raw_train_file)

        # train_dataset = QGDataset(train_dir, vocab, args.num_classes)
        train_dataset = NL2SPARQLDataset(pairs, vocab, MAX_LENGTH, device=device)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    print(train_dataset[0][0])
    for e in train_dataset[:10][0]:
        print(vocab.convertToLabels(e, 1000))
    # pairs = read_csv_file(file_test)

    # descomentar para usar el dataset filtrado
    # input_lang, output_lang, pairs = prepareData('spa.txt', filters=eng_prefixes, max_length=MAX_LENGTH)

    # print(random.choice(pairs))
    exit()
    similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes)
    # if args.sim == "cos":
    #     similarity = CosSimilarity(1)
    # else:
    #     similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes, dropout=True)

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        similarity,
        args.sparse)
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
