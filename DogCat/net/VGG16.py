#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： 实验二猫狗分类
@File    ：VGG16.py
@Author  ：郑家祥
@Date    ：2021/6/3 10:44 
@Description：复现经典网络VGG16的结构
'''
from torch import nn
from .BasicModule import BasicModule

class VGG16Block(nn.Module):
    """
    实现VGG16的子块
    """
    def __init__(self, inchannel, outchannel):
        super(VGG16Block, self).__init__()
        self.CNNBlock = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.CNNBlock(x)
        return out

class VGG16(BasicModule):
    """
    复现VGG16的网络结构，其包含多个layer，且每个layer都包含多个block
    """
    def __init__(self):
        super(VGG16, self).__init__()
        self.modelName = 'VGG16'
        self.layer1 = self._make_layer(3, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 3)
        self.layer4 = self._make_layer(256, 512, 3)
        self.layer5 = self._make_layer(512, 512, 3)
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.FC2 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.FC3 = nn.Sequential(
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x

    def _make_layer(self, inchannel, outchannel, block_num):
        '''
        构建layer
        '''
        layers = []
        layers.append(VGG16Block(inchannel, outchannel))
        for i in range(1, block_num):
            layers.append(VGG16Block(outchannel, outchannel))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
