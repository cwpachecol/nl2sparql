import os
import json
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import learning.transformers.constants as constants
from learning.transformers.tree import Tree
from learning.transformers.vocab import Vocab


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
