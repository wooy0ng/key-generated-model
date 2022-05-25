import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import random


class LSTMClassification(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, sequence_length, num_layers , device):
        super(LSTMClassification, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(
            input_size=self.in_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.criterion = nn.BCELoss()
        self.iteration = 0
        self.losses = []
    
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        self.context = out[:, -1, :]
        result = self.clf(self.context)
        return result

    def train(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.view(-1, 1).to(self.device)
        predicted = self.forward(inputs)
        
        loss = self.criterion(predicted, labels)
        self.losses.append(loss.item())
        self.iteration += 1
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class KeySimilarity(nn.Module):
    def __init__(self, args):
        super(KeySimilarity, self).__init__()
        self.in_size = args.hidden_size
        self.out_size = args.master_key_size
        self.embedding_size = 2

        self.in_size += self.embedding_size

        self.label_embedded = nn.Embedding(
            num_embeddings=2,
            embedding_dim=2
        )

        '''
        self.layer1 = nn.Sequential(
            nn.LayerNorm(self.in_size),
            nn.Linear(self.in_size, self.in_size),
            nn.LeakyReLU(),
            nn.Linear(self.in_size, self.in_size),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.in_size, self.out_size),
            nn.Sigmoid()
        )
        '''
        self.layer1 = nn.Sequential(
            nn.LayerNorm(self.in_size),
            nn.Linear(self.in_size, self.in_size),
            nn.PReLU(),
            nn.Linear(self.in_size, self.in_size),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.in_size, self.out_size),
            nn.Sigmoid()
        )
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)

        self.losses = []
        self.iteration = 0
    
    def forward(self, x):
        x = x.clone()
        out = self.layer1(x)
        out = out + x
        out = self.layer2(out)
        return out
    
    def train(self, inputs, labels, key):
        inputs = inputs.to(self.device)
        inputs = torch.cat((inputs, self.label_embedded(labels.long()).squeeze(1)), dim=1)

        labels = labels.to(self.device)
        outputs = self.forward(inputs)
        keys = key.repeat(outputs.shape[0], 1)
        for idx, label in enumerate(labels):
            if not label: 
                # keys[idx] = torch.rand(key.shape)
                # keys[idx] = torch.bitwise_xor(keys[idx].to(dtype=torch.int8), torch.ones(key.shape, dtype=torch.int8)).to(dtype=torch.float32)
                keys[idx] = torch.randint(0, 2, key.shape)

        loss = self.criterion(outputs, keys)
        self.iteration += 1
        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def test(self, inputs, labels):
        # labels : predicted labels
        inputs = torch.cat((inputs, self.label_embedded(labels.long()).squeeze(1)), dim=1)
        # inputs = torch.cat((inputs, self.label_embedded(torch.tensor([[1]])).squeeze(1)), dim=1)
        '''
        evaluate code
        inputs = torch.cat((inputs, self.label_embedded(torch.tensor([[1]])).squeeze(1)), dim=1)
        '''
        out = self.forward(inputs)
        return out