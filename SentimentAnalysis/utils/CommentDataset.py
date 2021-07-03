#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：CommentDataset.py
@Author  ：郑家祥
@Date    ：2021/6/23 23:34 
@Description：生成Dataset
'''
import torch
from torch.utils.data import Dataset
from zhconv import convert

class CommentDataset(Dataset):
    def __init__(self, file, word2id, id2word):
        self.file = file
        self.word2id = word2id
        self.id2word = id2word
        self.datas, self.labels = self.getboth()

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def getboth(self):
        datas, labels = [], []
        with open(self.file, encoding='utf-8') as f:
            for line in f.readlines():
                #取每行的label
                label = torch.tensor(int(line[0]), dtype=torch.int64)
                labels.append(label)
                #取每行的word
                line = convert(line, 'zh-cn')
                line_words = line.strip().split()[1:-1]
                indexs = []
                for word in line_words:
                    try:
                        index = self.word2id[word]
                    except BaseException:
                        index = 0
                    indexs.append(index)
                datas.append(indexs)
            return datas, labels

def mycollate_fn(data):
    #step1: 分离data、label
    data.sort(key=lambda x: len(x[0]), reverse=True)
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])

    #step2: 大于75截断、小于75补0
    padded_datas = []
    for data in input_data:
        if len(data) >= 75:
            padded_data = data[:75]
        else:
            padded_data = data
            while (len(padded_data) < 75):
                padded_data.append(0)
        padded_datas.append(padded_data)

    #step3: label、data转为tensor
    label_data = torch.tensor(label_data)
    padded_datas = torch.tensor(padded_datas, dtype=torch.int64)
    return padded_datas, label_data