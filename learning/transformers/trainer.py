from tqdm import tqdm

import torch
from torch.autograd import Variable as Var
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import utils
from learning.transformers.utils import map_label_to_target

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, epochs):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.epoch = 0

    # helper function for training
    def train(self, dataloader):

        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()



        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            # ltree, lsent, rtree, rsent, label = dataset[indices[idx]]
            # linput, rinput = Var(lsent), Var(rsent)
            # target = Var(map_label_to_target(label, dataset.num_classes))
            query, sparql = dataset[indices[idx]]
            if self.args.cuda:
                query, sparql = query.cuda(), sparql.cuda()
                # linput, rinput = linput.cuda(), rinput.cuda()
                # target = target.cuda()
            # output = self.model(ltree, linput, rtree, rinput)
            output = self.model(query, sparql)
            err = self.criterion(output, target)
            loss += err.data[0]
            err.backward()
            k += 1
            if k % self.args.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float)
        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            ltree, lsent, rtree, rsent, label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(output, target)
            loss += err.data
            output = output.data.squeeze().cpu()
            predictions[idx] = torch.dot(indices, torch.exp(output))
        return loss / len(dataset), predictions


class Trainer_Other(object):

    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        #self.device = args.device
        self.device = device
        self.epoch = 0
        # self.save_model = args.save_model
        self.exp_name = args.expname
        self.checkpoint_name = args.checkpoint_name

    def train(self, data_loader, question_field, sparql_field, epochs=5, last_checkpoint=0):
        self.model.to(self.device)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, verbose=True)

        # Tensorboard to get nice loss plot
        # writer = SummaryWriter("runs/loss_plot")
        # step = 0

        sentence = "Is Alexander Hamilton a lawyer?"
        bar = tqdm(range(last_checkpoint + 1, epochs + last_checkpoint + 1))
        for epoch in bar:
            print(f"[Epoch {epoch} / {epochs + last_checkpoint}]")
            self.model.train()
            losses = []

            for batch_idx, batch in enumerate(data_loader):
                # Get input and targets and get to cuda
                inp_data, target = batch.src, batch.trg
                inp_data, target = inp_data.to(self.device), target.to(self.device)

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

                # plot to tensorboard
                # writer.add_scalar("Training loss", loss, global_step=step)
                # step += 1

            mean_loss = sum(losses) / len(losses)
            print(f"loss mean: {(mean_loss)}")
            bar.set_description(f"loss {np.mean(losses):.5f}")

            scheduler.step(mean_loss)

            if self.args.save_model and (epoch + last_checkpoint) % 5 == 0:

                save_checkpoint_path = ""
                save_checkpoint_path = str(self.args.checkpoint_path) + str('/cp_') + str(self.args.checkpoint_name) + str('_') \
                                       + str(epoch) + str('_epochs.pth.tar')
                print(f"Saving checkpoint:{save_checkpoint_path}")
                checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
                utils.save_checkpoint(checkpoint, save_checkpoint_path)

            self.model.eval()
            translated_sentence = utils.translate_sentence(self.model, sentence, question_field, sparql_field, self.device, max_length=10000)
            print(f"Sentence: {sentence} \n")
            print(f"Translated example sentence: {translated_sentence} \n ")
        return mean_loss

    # helper function for training
    def train_old(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            print(dataset[indices[idx]])
            ltree, lsent, rtree, rsent, label = dataset[indices[idx]]
            linput, rinput = Var(lsent), Var(rsent)
            target = Var(map_label_to_target(label, dataset.num_classes))
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(output, target)
            loss += err.data[0]
            err.backward()
            k += 1
            if k % self.args.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss / len(dataset)

    # helper function for testing
    def test_old(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float)
        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            ltree, lsent, rtree, rsent, label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(output, target)
            loss += err.data
            output = output.data.squeeze().cpu()
            predictions[idx] = torch.dot(indices, torch.exp(output))
        return loss / len(dataset), predictions
