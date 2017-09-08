from __future__ import print_function

import os
import sys
import time
import fire

import numpy as np
np.random.seed(123)

import torch
torch.manual_seed(123)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

import common


# class LSTM(nn.Module):
#     def __init__(self, n_features, hidden_size, n_layers=1, use_cuda=True):
#         super(LSTM, self).__init__()
#         self.n_features = n_features
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.use_cuda = use_cuda

#         self.lstm = nn.RNN(n_features, hidden_size, n_layers, batch_first=True)
#         self.hidden2pred = nn.Linear(hidden_size, 1)

#     def init_state(self, batch_size):
#         # initial hidden state
#         h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
#         # initial cell state
#         c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
#         if self.use_cuda:
#             h0 = h0.cuda()
#             c0 = c0.cuda()

#         h0 = autograd.Variable(h0)
#         c0 = autograd.Variable(c0)
#         return h0 #, c0)

#     def forward(self, features):
#         batch_size = features.size(0)
#         initial_state = self.init_state(batch_size)
#         output, state = self.lstm(features, initial_state)
#         # Decode hidden state of last time step
#         pred = F.sigmoid(self.hidden2pred(output[:,-1,:]))
#         return pred

class RNN(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers=1, use_cuda=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        self.rnn = nn.RNN(n_features, hidden_size, n_layers, batch_first=True)
        self.hidden2pred = nn.Linear(hidden_size, 1)

    def forward(self, features):
        batch_size = features.size(0)
        initial_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            initial_state = initial_state.cuda()
        output, state = self.rnn(features, autograd.Variable(initial_state))
        pred = F.sigmoid(self.hidden2pred(output[:,-1,:]))
        return pred


def train(n_gpus=1, epochs=10, batch_size_per_gpu=128):

    device_ids = list(range(n_gpus)) 
    #device_ids = list(range(torch.cuda.device_count()))
    
    n_features = 29
    n_steps = 60
    hidden_size = 512
    batch_size = batch_size_per_gpu * max(1, n_gpus)
    use_cuda = n_gpus > 0

    print('n_gpus', n_gpus)
    print('device_ids', device_ids)
    print('batch_size_per_gpu', batch_size_per_gpu)
    print('batch_size_total', batch_size)

    x_train, y_train = common.load_test_data()

    # Remove last partial batch. It's not necessary in PyTorch,
    # but it's done here so that the timing comparisons against
    # Keras use the same number of points.
    n_batches = int(np.floor(len(x_train) / float(batch_size)))
    stop = n_batches * batch_size
    print('stop', stop)
    x_train = x_train[:stop]
    y_train = y_train[:stop]

    model = RNN(n_features, hidden_size, use_cuda=use_cuda)

    print('-'*50)
    for w in model.parameters():
        print(w.size())

    n_pars = sum([np.prod(w.size()) for w in model.parameters()])
    print('Number of parameters: {:,}'.format(n_pars))

    if n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if use_cuda:
        torch.cuda.manual_seed(123)
        model = model.cuda()

    criterion = nn.MSELoss()
    if use_cuda:
        criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_indices = common.make_batches(len(x_train), batch_size)
    nb_batches = len(batch_indices)

    with open('pytorch_lstm_x%i.csv' % n_gpus, 'w+') as f:
        f.write('epoch, epoch_time, loss\n')

        for epoch in range(epochs):
            train_loss = 0.
            epoch_time = time.time()

            for batch_num, (start, stop) in enumerate(batch_indices):
                model.zero_grad()

                x_batch = torch.Tensor(x_train[start:stop])
                y_batch = torch.Tensor(y_train[start:stop])

                if use_cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                x_batch = autograd.Variable(x_batch)
                y_batch = autograd.Variable(y_batch)

                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]

            train_loss = train_loss/(batch_num+1)
            epoch_time = time.time() - epoch_time

            print('epoch: %i, time: %.2f, loss: %.4f' \
                % (epoch, epoch_time, train_loss))
            
            f.write('%i, %.4f, %.4f\n' % (epoch, epoch_time, train_loss))


if __name__ == '__main__':
    fire.Fire()

"""
orch.Size([4, 29])
torch.Size([4, 4])
torch.Size([4])
torch.Size([4])
"""
