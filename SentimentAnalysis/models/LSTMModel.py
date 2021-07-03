#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：LSTMModel.py
@Author  ：郑家祥
@Date    ：2021/6/24 13:05 
@Description：构建LSTM模型
'''
import torch
from torch import nn
from .BasicModule import BasicModule

class LSTMModel(BasicModule):
    def __init__(self, embedding_dim, hidden_dim, pre_weight):
        super(LSTMModel, self).__init__()
        self.modelName = 'LSTMModel'
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        self.embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3, batch_first=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, input, hidden = None):
        '''
        input的size为(batch_size, max_len)
        '''
        batch_size, max_len = input.size()
        embeds = self.embeddings(input)
        embeds = torch.tensor(embeds, dtype=torch.float32)
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output)
        #取最后一个时间步的输出
        last_outputs = self.get_last_output(output, batch_size, max_len)
        return last_outputs, hidden

    def get_last_output(self, output, batch_size, max_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(batch_size):
            last_outputs[i] = output[i][max_len - 1]
        last_outputs = last_outputs.to(output.device)
        return last_outputs