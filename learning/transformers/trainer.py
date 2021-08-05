from tqdm import tqdm

import torch
from torch.autograd import Variable as Var
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import utils
from learning.transformers.utils import map_label_to_target

class Trainer(object):
    def __init__(self, args, model, device, criterion, optimizer, epochs):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.epoch = 0
        self.model.to(self.device)

    # helper function for training
    def train(self, dataloader):
        self.model.train()
        losses = []
        bar = tqdm(dataloader['train'])
        for batch_idx, batch in enumerate(bar):
            # for batch_idx, batch in enumerate(dataloader['train']):
            # Get input and targets and get to cuda
            inp_data, target = batch
            inp_data = inp_data.to(self.device)
            target = target.to(self.device)

            # Forward prop
            output = self.model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            self.optimizer.zero_grad()

            loss = self.criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # Gradient descent step
            self.optimizer.step()

        mean_loss = sum(losses) / len(losses)
        # print(f"loss mean: {(mean_loss)}")
        return mean_loss

    # helper function for testing
    def test(self, dataloader, vocab_s, max_length):
        val_loss = []
        self.model.eval()
        with torch.no_grad():
            bar = tqdm(dataloader['test'])
            for batch in bar:
                input_sentences, output_sentences = batch
                outputs = []
                outputs.insert(0, vocab_s.getIndex("<bos>"))
                outputs.extend(input_sentences.unsqueeze(1))
                outputs.append(vocab_s.getIndex("<eos>"))
                print(outputs)
                outputs = torch.tensor(outputs)

                # bs = input_sentences.shape[0]
                # outputs = torch.tensor([vocab_s.getIndex("<bos>") for b in range(bs)], device=self.device)
                print(outputs)
                print("^"*20)
                for i in range(max_length):
                    # trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(self.device)
                    trg_tensor = outputs.to(self.device)
                    # trg_tensor = outputs.unsqueeze(1).to(self.device)
                    with torch.no_grad():
                        output = self.model(outputs, trg_tensor)

                    best_guess = output.argmax(2)[-1, :].item()
                    outputs.append(best_guess)

                    if best_guess == vocab_s.getIndex("<eos>"):
                        break

                translated_sentence = [vocab_s.getLabel(idx) for idx in outputs]