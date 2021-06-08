#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：深度学习
@File    ：实验一手写体识别.py
@Author  ：郑家祥_202028019410032
@Date    ：2021/5/17 21:57
@Description：利用MNIST数据集在Pytorch框架上进行手写数字体识别，测试误差达到98%及以上
"""

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import torch.utils.data as Data
from tensorboardX import SummaryWriter
import hiddenlayer as hl

def createBatchData(batch_size, resize=None):
    '''
    使用DataLoader创建随机批量数据
    '''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root='data', train=True, transform=trans, download=False)
    minst_test = torchvision.datasets.MNIST(root='data', train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle = True, num_workers=4),
            data.DataLoader(minst_test, batch_size, shuffle = False, num_workers=4))

class myNet(nn.Module):
    '''
    构建卷积神经网络,与LeNet类似
    '''
    def __init__(self):
        super(myNet, self).__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),#6*28*28
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2)#6*14*14
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1), #16*10*10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2) #16*5*5
        )
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def reshape_(self, x):
        return x.reshape(-1, 1, 28, 28)

    def forward(self, x):
        x = self.reshape_(x)
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.FC1(x)
        return x

def sameNumber(y_hat, y):
    '''
    返回预测值与真实值相等的个数
    '''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator():
    '''
    构建n列变量，每列累加
    '''
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def testNet(net, testData):
    '''
    计算测试误差
    '''
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in testData:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        sameNum = sameNumber(y_hat, y)
        metric.add(sameNum, y.numel())
    return metric[0] / metric[1]

def trainNet(net, trainData, testData, num_epochs, lr, device):
    '''
    使用trainData训练网络
    '''
    def init(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    lossFun = nn.CrossEntropyLoss()
    writer = SummaryWriter(logdir='assets/visualize', comment="test1")
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(trainData):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = lossFun(y_hat, y)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], sameNumber(y_hat, y), X.shape[0])
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = testNet(net, testData)
        writer.add_scalar('TrainLoss', train_loss, epoch)
        writer.add_scalar('TrainAccuracy', train_acc, epoch)
        writer.add_scalar('TestAccuracy', test_acc, epoch)
        print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')

if __name__ == '__main__':
    batch_size = 256
    num_epochs = 100
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    trainDataIter, testDataIter = createBatchData(batch_size);

    net = myNet()
    '''
    modelVis = hl.build_graph(net, torch.zeros([1 ,1, 28, 28]))
    modelVis.theme = hl.graph.THEMES["blue"].copy()
    modelVis.save("assets/test2",format='png')
    '''
    trainNet(net, trainDataIter, testDataIter, num_epochs=num_epochs, lr=lr, device=device);
