import os
import json
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import learning.transformers.constants as constants
from learning.transformers.tree import Tree
from learning.transformers.vocab import Vocab


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS, EOS and PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]

    def sentenceFromIndex(self, index):
        return [self.index2word[ix] for ix in index]

class NL2SPARQLDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, vocab, max_length, device):
        # self.input_lang = input_lang
        # self.output_lang = output_lang
        self.pairs = pairs
        self.vocab = vocab
        self.max_length = max_length
        self.device = device
        self.queries = self.read_queries(self.pairs)
        self.sparqls = self.read_sparqls(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        query = deepcopy(self.queries[index])
        sparql = deepcopy(self.sparqls[index])
        return (query, sparql)

    def read_queries(self, pairs):
        queries = [self.read_sentence(row[0]) for row in tqdm(pairs)]
        return queries

    def read_sparqls(self, pairs):
        sparqls = [self.read_sentence(row[1]) for row in tqdm(pairs)]
        return sparqls

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), constants.UNK_WORD)
        return torch.LongTensor(indices)

class QGDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(QGDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.lsentences = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))

        self.ltrees = self.read_trees(os.path.join(path, 'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path, 'b.parents'))

        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = len(self.lsentences)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (ltree, lsent, rtree, rsent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        # print("+"*20)
        # print(f"filename: {filename}")
        # print("#" * 20)
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        # print("+" * 20)
        # print(f"parents: {parents}")
        # print("+" * 20)
        trees = dict()
        root = None
        # print("*" * 30)
        # print(f"len(parents): {len(parents)}")
        # print("*"*30)
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                # print("=" * 40)
                while True:

                    # print(idx)
                    # print(parents[:])
                    # print(parents[idx - 1])
                    if(len(parents) < idx):
                        parent = parents[idx - 1]
                        if parent == -1:
                            break
                    else:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels
