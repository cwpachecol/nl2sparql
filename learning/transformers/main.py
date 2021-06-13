from __future__ import division
from __future__ import print_function

import os
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
from learning.transformers.model import *
# DATA HANDLING CLASSES
from tree import Tree
from learning.transformers.vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from learning.transformers.dataset import QGDataset
# METRICS CLASS FOR EVALUATION
from learning.transformers.metrics import Metrics
# UTILITY FUNCTIONS
from learning.transformers.utils import load_word_vectors, build_vocab
# CONFIG PARSER
from learning.transformers.config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from learning.transformers.trainer import Trainer
import datetime
from fasttext import load_model
import spacy
# from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from generator_utils import decode, fix_URI

sys.path.insert(0, os.path.abspath("..//.."))

spacy_eng = spacy.load("en_core_web_sm")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

tokenize = lambda x: x.split()

def load_dataset(path_datasets, train_file, test_file, valid_file, batch_size=1, device='cpu'):
    question_field = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
    sparql_field = Field(tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>")
    fields = {'question': ('src', question_field), 'query': ('trg', sparql_field)}
    train_data, valid_data, test_data = TabularDataset.splits(path=path_datasets,
                                                              train=train_file,
                                                              test=test_file,
                                                              validation=valid_file,
                                                              format="json",
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
    save_dir = base_dir + '\\' + args.save

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
    # args.data = 'learning/transformers/data/lcquad10/'
    # args.save = 'learning/transformers/checkpoints/'

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.benchmark = True
    # if not os.path.exists(base_dir + '\\' + args.save):
    #     os.makedirs(base_dir + '\\' + args.save)
    #
    # train_dir = os.path.join(base_dir + '\\' + args.data, 'train/')
    # dev_dir = os.path.join(base_dir + '\\' + args.data, 'dev/')
    # test_dir = os.path.join(base_dir + '\\' + args.data, 'test/')
    #
    # # write unique words from all token files
    # dataset_vocab_file = os.path.join(base_dir + '\\' + args.data, 'dataset.vocab')

    # token_files_a = []
    # for split in [train_dir, dev_dir, test_dir]:
    #     print(split)
    #     token_files_a.extend([os.path.join(split, 'a.toks')])
    # print(token_files_a)
    # exit()

    # if not os.path.isfile(dataset_vocab_file):
    #     token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
    #     token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
    #
    #     token_files = token_files_a + token_files_b
    #     dataset_vocab_file = os.path.join(base_dir + '\\' + args.data, 'dataset.vocab')
    #     build_vocab(token_files, dataset_vocab_file)
    #
    # # get vocab object from vocab file previously written
    # vocab = Vocab(filename=dataset_vocab_file,
    #               data=[constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD])
    # logger.debug('==> Dataset vocabulary size : %d ' % vocab.size())
    #
    # # load dataset splits
    # train_file = os.path.join(base_dir + '\\' + args.data, 'dataset_train.pth')
    # if os.path.isfile(train_file):
    #     train_dataset = torch.load(train_file)
    # else:
    #     # print(train_dir)
    #     # print(vocab)
    #     # print(args.num_classes)
    #     train_dataset = QGDataset(train_dir, vocab, args.num_classes)
    #     print(train_dataset)
    #     torch.save(train_dataset, train_file)
    # logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    #
    # dev_file = os.path.join(base_dir + '\\' + args.data, 'dataset_dev.pth')
    # if os.path.isfile(dev_file):
    #     dev_dataset = torch.load(dev_file)
    # else:
    #     dev_dataset = QGDataset(dev_dir, vocab, args.num_classes)
    #     torch.save(dev_dataset, dev_file)
    # logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    #
    # test_file = os.path.join(base_dir + '\\' + args.data, 'dataset_test.pth')
    # if os.path.isfile(test_file):
    #     test_dataset = torch.load(test_file)
    # else:
    #     test_dataset = QGDataset(test_dir, vocab, args.num_classes)
    #     torch.save(test_dataset, test_file)
    # logger.debug('==> Size of test data    : %d ' % len(test_dataset))
    #
    # similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes)
    # if args.sim == "cos":
    #     similarity = CosSimilarity(1)
    # else:
    #     similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes, dropout=True)

    # initialize model, criterion/loss_function, optimizer

    # model = SimilarityTreeLSTM(
    #     vocab.size(),
    #     args.input_dim,
    #     args.mem_dim,
    #     similarity,
    #     args.sparse)

    dataset_path = base_dir + '\\' + 'learning/transformers/data/lcquad10'
    train_file = base_dir + '\\' + 'learning/transformers/data/lcquad10/LCQuad10_train.json'
    test_file = base_dir + '\\' + 'learning/transformers/data/lcquad10/LCQuad10_test.json'
    valid_file = base_dir + '\\' + 'learning/transformers/data/lcquad10/LCQuad10_trial.json'
    batch_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing in: {device}")

    train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data, question_field, sparql_field = load_dataset(
        dataset_path, train_file, test_file, valid_file, batch_size, device)
    src_vocab_size = len(question_field.vocab)
    trg_vocab_size = len(sparql_field.vocab)

    print(src_vocab_size, trg_vocab_size)
    print("-"*10)
    # Model hyperparameters
    embedding_size = 256
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 1010
    forward_expansion = 4
    src_pad_idx = question_field.vocab.stoi["<pad>"]
    last_checkpoint = 0
    learning_rate = 3e-4
    # last_checkpoint = args_from_file['last_checkpoint']

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
    )

    # criterion = nn.KLDivLoss()  # nn.HingeEmbeddingLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # if args.cuda:
    #     model.cuda(), criterion.cuda()
    # else:
    #     torch.set_num_threads(4)
    # logger.info("number of available cores: {}".format(torch.get_num_threads()))
    # if args.optim == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # elif args.optim == 'adagrad':
    #     optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # elif args.optim == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # metrics = Metrics(args.num_classes)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)

    # # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # # for other words in dataset vocab, use random normal vectors
    # emb_file = os.path.join(base_dir + '\\' + args.data, 'dataset_embed.pth')
    # if os.path.isfile(emb_file):
    #     emb = torch.load(emb_file)
    # else:
    #     EMBEDDING_DIM = 300
    #     emb = torch.zeros(vocab.size(), EMBEDDING_DIM, dtype=torch.float)
    #     fasttext_model = load_model(base_dir + '\\' + "data/fasttext/wiki.en.bin")
    #     print('Use Fasttext Embedding')
    #     for word in vocab.labelToIdx.keys():
    #         word_vector = fasttext_model.get_word_vector(word)
    #         if word_vector.all() != None and len(word_vector) == EMBEDDING_DIM:
    #             emb[vocab.getIndex(word)] = torch.Tensor(word_vector)
    #         else:
    #             emb[vocab.getIndex(word)] = torch.Tensor(EMBEDDING_DIM).uniform_(-1, 1)
    #     # # load glove embeddings and vocab
    #     # args.glove = 'learning/transformers/data/glove/'
    #     # print('Use Glove Embedding')
    #     # glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
    #     # logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
    #     # emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(-0.05, 0.05)
    #     # # zero out the embeddings for padding and other special words if they are absent in vocab
    #     # for idx, item in enumerate([constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD]):
    #     #     emb[idx].zero_()
    #     # for word in vocab.labelToIdx.keys():
    #     #     if glove_vocab.getIndex(word):
    #     #         emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
    #     torch.save(emb, emb_file)
    # # plug these into embedding matrix inside model
    # if args.cuda:
    #     emb = emb.cuda()
    # model.emb.weight.data.copy_(emb)

    # checkpoint_filename = '%s.pt' % os.path.join(base_dir + '\\' + args.save, args.expname)
    # if args.mode == "test":
    #     checkpoint = torch.load(checkpoint_filename)
    #     model.load_state_dict(checkpoint['model'])
    #     args.epochs = 1

    # create trainer object for training and testing

    # trainer = Trainer(args, model, criterion, optimizer)
    # logger.debug(f"==> Trainer parameters (args)     : { args }")
    #logger.debug(f"==> Trainer parameters (model)     : {model}")
    # logger.debug(f"==> Trainer parameters (criterion)     : {criterion}")
    # logger.debug(f"==> Trainer parameters (optimizer)     : {optimizer}")

    # Prepare path's and files
    # dataset_name = args_from_file['dataset_name']
    # dataset_path = args_from_file['dataset_path'] + '/' + dataset_name
    # checkpoint_path = args_from_file['checkpoint_path'] + '/' + dataset_name


    if args.mode == "train":
        trainer = Trainer(args, model, criterion, optimizer, device)
        train_loss = trainer.train(train_iterator, 10, 0)
        print(f"train loss: { train_loss }")

    exit()
    # for epoch in range(args.epochs):
    #     if args.mode == "train":
    #         # scheduler.step()
    #         train_loss = trainer.train(train_dataset)
    #         exit()
    #         train_loss, train_pred = trainer.test(train_dataset)
    #         logger.info(
    #             '==> Epoch {}, Train \tLoss: {} {}'.format(epoch, train_loss,
    #                                                        metrics.all(train_pred, train_dataset.labels)))
    #         checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
    #                       'args': args, 'epoch': epoch, 'scheduler': scheduler}
    #         checkpoint_filename = '%s.pt' % os.path.join(args.save,
    #                                                      args.expname + ',epoch={},train_loss={}'.format(epoch + 1,
    #                                                                                                    train_loss))
    #         torch.save(checkpoint, checkpoint_filename)
    #         exit()
    #     dev_loss, dev_pred = trainer.test(dev_dataset)
    #     test_loss, test_pred = trainer.test(test_dataset)
    #     logger.info(
    #         '==> Epoch {}, Dev \tLoss: {} {}'.format(epoch, dev_loss, metrics.all(dev_pred, dev_dataset.labels)))
    #     logger.info(
    #         '==> Epoch {}, Test \tLoss: {} {}'.format(epoch, test_loss, metrics.all(test_pred, test_dataset.labels)))


if __name__ == "__main__":
    main()
