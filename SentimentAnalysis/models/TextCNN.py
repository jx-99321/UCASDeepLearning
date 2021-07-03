#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：TextCNN.py
@Author  ：郑家祥
@Date    ：2021/6/24 13:06 
@Description：构建TextCNN模型
'''
import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class TextCNN(BasicModule):
    def __init__(self, vocab_size, embedding_dim, filters_num, filter_size, pre_weight):
        super(TextCNN, self).__init__()
        self.modelName = 'TextCNN'
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.requires_grad = False
        if pre_weight is not None:
            self.embeddings = self.embeddings.from_pretrained(pre_weight)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filters_num, (size, embedding_dim)) for size in filter_size])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filters_num * len(filter_size), 2)

    def forward(self, x):
        '''
        x的size为(batch_size, max_len)
        '''
        x = self.embeddings(x)  #(batch_size, max_len, embedding_dim)
        x = x.unsqueeze(1)      #(batch_size, 1, max_len, embedding_dim)
        x = torch.tensor(x, dtype=torch.float32)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out