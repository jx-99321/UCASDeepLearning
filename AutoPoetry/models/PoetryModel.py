#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： AutoPoetry
@File    ：PoetryModel.py
@Author  ：郑家祥
@Date    ：2021/6/16 22:18 
@Description：
'''
import torch.nn as nn
from .BasicModule import BasicModule

class PoetryModel(BasicModule):
    """
    描述：自定义循环神经网络，包括embedding、LSTM、FC_layer
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.modelName = 'PoetryModel'
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, vocab_size)
        )

    def forward(self, input, hidden = None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output.view(seq_len * batch_size, -1))
        return output, hidden

