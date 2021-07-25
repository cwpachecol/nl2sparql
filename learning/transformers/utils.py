# from __future__ import division
# from __future__ import print_function

import os
import math

import torch
# import spacy
from torchtext.data.metrics import bleu_score
import sys

from learning.treelstm.tree import Tree
from learning.treelstm.vocab import Vocab


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path + '.txt', encoding='utf-8', errors='ignore'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * (count)
    vectors = torch.zeros(count, dim)
    with open(path + '.txt', 'r', encoding='utf-8', errors='ignore') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path + '.vocab', 'w', encoding='utf-8', errors='ignore') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    tokens = line.rstrip('\n').split(' ')
                    vocab |= set(tokens)
    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes)
    # if label == -1:
    #     target[0][0] = 1
    # else:
    #     target[0][1] = 1
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor - 1] = 1
    else:
        target[0][floor - 1] = ceil - label
        target[0][ceil - 1] = label - floor
    return target
import torch

def lreplace(pattern, sub, string):
    """
    Replaces "pattern" in "string" with "sub" if "pattern" starts "string".
    """
    return re.sub("^%s" % pattern, sub, string)

def rreplace(pattern, sub, string):
    """
    Replaces "pattern" in "string" with "sub" if "pattern" ends "string".
    """
    return re.sub("%s$" % pattern, sub, string)



def translate_sentence(model, sentence, question, sparql, device, max_length=50):
    # Load question tokenizer
    tokenize = lambda x: x.split()

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.lower() for token in tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, question.init_token)
    tokens.append(question.eos_token)

    # Go through each question token and convert to an index
    text_to_indices = [question.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [sparql.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == sparql.vocab.stoi["<eos>"]:
            break

    translated_sentence = [sparql.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, question, sparql, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, question, sparql, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, device):
    print("=> Loading checkpoint")

    if device == "cpu":
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
