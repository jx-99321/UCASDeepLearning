#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： AutoPoetry
@File    ：PoetryData.py
@Author  ：郑家祥
@Date    ：2021/6/16 15:34 
@Description：用于生成数据，包括data、ix2word、word2ix
'''
import numpy as np
import torch
from torch.utils.data import DataLoader

def poetryData(filename, batch_size):
    """
    描述：从npz文件中获取data、ix2word、word2ix，其中ix2word序号到字的映射，word2ix为字到序号的映射
    """
    #step1: 读取数据
    dataset = np.load(filename, allow_pickle=True)
    data = dataset['data']
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()

    #step2: 转为tensor并输出
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader, ix2word, word2ix

if __name__ == "__main__":
    dataloader, ix2word, word2ix = poetryData(r'..\data\tang.npz',16)
    print("ix2word: ", len(ix2word))
    print("word2ix: ", len(word2ix))
    for i, data in enumerate(dataloader):
        if i ==1 :
            for m in range(0,16):
                result = []
                for n in range(0, 125):
                    index = data[m][n].item()
                    w = ix2word[index]
                    result.append(w)
                print(result)

