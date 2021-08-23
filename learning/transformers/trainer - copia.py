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
    def train(self, train_iterator):
        self.model.train()
        losses = []
        # bar = tqdm(train_iterator)
        # for batch in bar:
        for batch_idx, batch in enumerate(train_iterator):
            # for batch_idx, batch in enumerate(dataloader['train']):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(self.device)
            target = batch.trg.to(self.device)

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
            print("-"*20)
            print(loss.item())
            print("-" * 20)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # Gradient descent step
            self.optimizer.step()

        mean_loss = sum(losses) / len(losses)
        print(f"loss mean: {(mean_loss)}")
        return mean_loss

    # helper function for testing
    def test(self, test_iterator, max_length):
        val_loss = []
        self.model.eval()
        losses = []
        outputs = []
        with torch.no_grad():

            bar = tqdm(test_iterator)
            for batch in bar:
                inp_data = batch.src.to(self.device)
                target = batch.trg.to(self.device)

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
                loss = self.criterion(output, target)
                losses.append(loss.item())
                outputs.append(output)
        mean_loss = sum(losses) / len(losses)
        print(f"loss mean: {(mean_loss)}")
        return mean_loss, outputs

