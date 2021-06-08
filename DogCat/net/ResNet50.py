#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：实验二猫狗分类
@File    ：ResNet50.py
@Author  ：郑家祥
@Date    ：2021/6/3 13:31 
@Description：复现经典网络ResNet50的结构
'''
from torch import nn
from .BasicModule import BasicModule

class ResNet50Block(nn.Module):
    """
       实现ResNet50的子块
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=False):
        super(ResNet50Block, self).__init__()
        self.CNNBlock = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel * 4),
        )
        self.relu =nn.ReLU(inplace=True)
        self.shortcut = shortcut
        if self.shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * 4)
            )

    def forward(self, x):
        if self.shortcut:
            residual = self.shortcut(x)
        else:
            residual = x
        x = self.CNNBlock(x)
        x += residual
        x = self.relu(x)
        return x

class ResNet50(BasicModule):
    """
    复现ResNet50的网络结构，其包含多个layer，且每个layer都包含多个block
    """
    def __init__(self):
        super(ResNet50, self).__init__()
        self.modelName = 'ResNet50'
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, 3, 1)
        self.layer2 = self._make_layer(256, 128, 4, 2)
        self.layer3 = self._make_layer(512, 256, 6, 2)
        self.layer4 = self._make_layer(1024, 512, 3, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.FC(x)
        return x

    def _make_layer(self, inchannel, outchannel, block_num, stride):
        '''
        构建layer
        '''
        layers = []
        layers.append(ResNet50Block(inchannel, outchannel, stride, shortcut=True))
        for i in range(1, block_num):
            layers.append(ResNet50Block(outchannel * 4, outchannel))
        return nn.Sequential(*layers)

