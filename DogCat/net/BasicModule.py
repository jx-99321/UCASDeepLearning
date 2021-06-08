#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：实验二猫狗分类
@File    ：BasicModule.py
@Author  ：郑家祥
@Date    ：2021/6/3 10:43 
@Description：BasicModule是对nn.Module的简易封装，提供快速加载和保存模型的接口
'''
import torch
from torch import nn
import time

class BasicModule(nn.Module):
    """
    封装nn.Module，提供load模型和save模型接口
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.modelName = str(type(self))

    def load(self, path):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存训练的模型到指定路径
        '''
        if name is None:
            prepath = 'models/' + self.modelName + '_'
            name = time.strftime(prepath + '%m%d_%H_%M.pth')
        torch.save(self.state_dict(), name)
        print("保存的模型路径为：", name)
        return name


