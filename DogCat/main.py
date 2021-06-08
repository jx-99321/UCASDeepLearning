#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：实验二猫狗分类
@File    ：main.py
@Author  ：郑家祥
@Date    ：2021/5/17 21:57
@Description：
'''

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from data.CatDog import CatDog
from net import VGG16, ResNet50
import time
import torchvision.models as models

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
    构建n列变量，每列累加，便于计算准确率与损失
    '''
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def train(net, batch_size, lr, epochs, device, trainwriter, validatewriter, pre_model_path=None):
    """
    描述：训练模型并使用验证数据集验证
    """
    # step1: 模型初始化
    if pre_model_path:
        net.load(pre_model_path)
    else:
        def init(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)
        net.apply(init)
    net.to(device)


    #step2: 训练数据与验证数据
    data = CatDog(root=r'data\train', train=True)
    train_data, validate_data = Data.random_split(data, [int(0.7 * len(data)), int(0.3 * len(data))])
    train_loader = Data.DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    validate_loader = Data.DataLoader(validate_data, batch_size, shuffle=True, num_workers=4)

    #step3: 定义目标函数与优化器，规定学习率衰减规则
    lossFun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    #step4: 训练，计算训练损失、准确率、验证损失、准确率
    metric = Accumulator(3)
    for epoch in range(epochs):
        net.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = lossFun(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], sameNumber(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            trainwriter.add_scalar('Batch Loss', train_loss, (epoch * (len(train_data)/batch_size) + i))
            if i % 50 == 0:
                print('epoch: {:d}, batch: {:d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, i, train_loss, train_acc))
        scheduler.step(train_loss)
        trainwriter.add_scalar('Epoch Loss', train_loss, epoch)
        trainwriter.add_scalar('Accuracy', train_acc, epoch)

        #验证，计算验证损失和准确率
        validate_loss, validate_acc = validate(net, validate_loader, lossFun)
        print('epoch: {:d}, Validate Loss: {:.4f}, Validate Accuracy: {:.4f}'.format(epoch, validate_loss, validate_acc))
        validatewriter.add_scalar('Epoch Loss', validate_loss, epoch)
        validatewriter.add_scalar('Accuracy', validate_acc, epoch)

        #当验证准确度达到95%以上时保存模型
        if validate_acc > 0.95:
            net.save()
            print("该模型是在第{}个epoch取得95%以上的验证准确率, 准确率为：{:.4f}".format(epoch, validate_acc))

    #step5: 返回迭代完的模型用于测试
    trainwriter.close()
    validatewriter.close()
    return net


def validate(net, data_loader, lossFun):
    """
    描述：验证模型的准确率并计算损失
    """
    net.eval()
    metric = Accumulator(3)
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = lossFun(y_hat, y)
        metric.add(loss * X.shape[0], sameNumber(y_hat, y), X.shape[0])
        validate_loss = metric[0] / metric[2]
        validate_acc = metric[1] / metric[2]
    net.train()
    return validate_loss, validate_acc

def test(net, batch_size, device, testwriter):
    """
    描述：测试模型的准确率并计算损失
    """
    #step1: 测试数据
    test_data = CatDog(root=r'data\test', train=False)
    test_loader = Data.DataLoader(test_data, batch_size, shuffle=True, num_workers=4)

    #step2: 定义目标函数
    lossFun = nn.CrossEntropyLoss()

    #step3: 测试，计算测试损失和准确率
    metric = Accumulator(3)
    net.to(device)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = lossFun(y_hat, y)
        metric.add(loss * X.shape[0], sameNumber(y_hat, y), X.shape[0])
        test_loss = metric[0] / metric[2]
        test_acc = metric[1] / metric[2]
        testwriter.add_scalar('Loss', test_loss, i)
        testwriter.add_scalar('Accuracy', test_acc, i)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))

if __name__ == "__main__":

    batch_size = 10
    lr = 0.01
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    #使用resnet50训练
    net = ResNet50()
    visdir = time.strftime( 'assets/visualize/' + net.modelName + '_%m%d_%H_%M')
    trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    validatewriter = SummaryWriter('{}/{}'.format(visdir, 'Validate'))
    testwriter = SummaryWriter('{}/{}'.format(visdir, 'Test'))
    trainedNet = train(net, batch_size, lr, epochs, device, trainwriter, validatewriter)
    test(trainedNet, batch_size, device, testwriter)

    '''
    #使用resnet50预训练模型训练并测试
    resnet50 = models.resnet50(pretrained=True)
    numFit = resnet50.fc.in_features
    resnet50.fc = nn.Linear(numFit, 2)
    visdir = time.strftime( 'assets/visualize/' + 'ResNet50_pretrained' + '_%m%d_%H_%M')
    trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    validatewriter = SummaryWriter('{}/{}'.format(visdir, 'Validate'))
    testwriter = SummaryWriter('{}/{}'.format(visdir, 'Test'))
    trainedNet = train(resnet50, batch_size, lr, epochs, device, trainwriter, validatewriter)
    test(trainedNet, batch_size, device, testwriter)
    '''

    '''
    #使用vgg16训练
    net = VGG16()
    visdir = time.strftime('assets/visualize/' + net.modelName + '_%m%d_%H_%M')
    trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    validatewriter = SummaryWriter('{}/{}'.format(visdir, 'Validate'))
    testwriter = SummaryWriter('{}/{}'.format(visdir, 'Test'))
    trainedNet = train(net, batch_size, lr, epochs, device, trainwriter, validatewriter)
    test(trainedNet, batch_size, device, testwriter)
    '''

    '''
    #使用vgg预训练模型只更新全连接层参数训练并测试
    vgg16 = models.vgg16(pretrained=True)
    for parameter in vgg16.parameters():
        parameter.requires_grad = False
    vgg16.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, 2),
    )
    visdir = time.strftime('assets/visualize/' + 'VGG16_pretrained' + '_%m%d_%H_%M')
    trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    validatewriter = SummaryWriter('{}/{}'.format(visdir, 'Validate'))
    testwriter = SummaryWriter('{}/{}'.format(visdir, 'Test'))
    trainedNet = train(vgg16, batch_size, lr, epochs, device, trainwriter, validatewriter)
    test(trainedNet, batch_size, device, testwriter)
    '''