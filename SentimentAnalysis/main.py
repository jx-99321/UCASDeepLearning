#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：main.py
@Author  ：郑家祥
@Date    ：2021/6/23 11:32 
@Description：实现模型的训练、验证与测试
'''
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from utils.CommentDataset import CommentDataset, mycollate_fn
from models import LSTMModel, TextCNN
from utils.BuildData import build_word2id, build_word2vec
import time

class Accumulator():
    """
    构建n列变量，每列累加，便于计算准确率与损失
    """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def sameNumber(y_hat, y):
    """
    返回预测值与真实值相等的个数
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()

def train(model, batch_size, lr, epochs, device, trainwriter, validatewriter, trainloader, validateloader):
    """
    描述：训练模型并计算损失、准确率
    """
    #step1: 将模型设置到device上
    model.to(device)
    model.train()

    #step2: 定义目标函数与优化器，规定学习率衰减规则
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

    #step3: 训练模型计算损失、准确率并在每个epoch进行验证
    metric = Accumulator(3)
    step = 0
    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            if model.modelName == 'LSTMModel':
                outputs, hidden = model(inputs)
            elif model.modelName == 'TextCNN':
                outputs = model(inputs)
            #print(outputs, labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            with torch.no_grad():
                metric.add(loss * inputs.shape[0], sameNumber(outputs, labels), inputs.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            trainwriter.add_scalar('Batch Loss', train_loss, step)
            if i % 50 == 0:
                print('epoch: {:d}, batch: {:d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, i, train_loss, train_acc))
        scheduler.step(train_loss)
        trainwriter.add_scalar('Epoch Loss', train_loss, epoch)
        trainwriter.add_scalar('Accuracy', train_acc, epoch)

        # 验证，计算验证损失和准确率
        validate_loss, validate_acc = validate(model, validateloader, criterion)
        print('epoch: {:d}, Validate Loss: {:.4f}, Validate Accuracy: {:.4f}'.format(epoch, validate_loss, validate_acc))
        validatewriter.add_scalar('Epoch Loss', validate_loss, epoch)
        validatewriter.add_scalar('Accuracy', validate_acc, epoch)

        # 当验证准确度达到83%以上时保存模型
        if train_acc >0.80 and validate_acc > 0.80 :
            model.save()
            print("该模型是在第{}个epoch取得80%以上的验证准确率, 准确率为：{:.4f}".format(epoch, validate_acc))

    #step4: 迭代结束保存模型
    trainwriter.close()
    validatewriter.close()

    #step5: 返回模型用于测试
    return model

def validate(model, validateloader, criterion):
    """
    描述：验证模型的准确率并计算损失
    """
    model.eval()
    metric = Accumulator(3)
    for i, data in enumerate(validateloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        if model.modelName == 'LSTMModel':
            outputs, hidden = model(inputs)
        elif model.modelName == 'TextCNN':
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        metric.add(loss * inputs.shape[0], sameNumber(outputs, labels), inputs.shape[0])
        validate_loss = metric[0] / metric[2]
        validate_acc = metric[1] / metric[2]
    model.train()
    return validate_loss, validate_acc

def test(model, device, testloader, testwriter):
    """
    描述：测试模型的准确率并计算损失
    """
    #step1: 定义目标函数
    criterion = nn.CrossEntropyLoss()

    #step2: 测试，计算测试损失和准确率
    metric = Accumulator(3)
    model.to(device)
    for i, data in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        if model.modelName == 'LSTMModel':
            outputs, hidden = model(inputs)
        elif model.modelName == 'TextCNN':
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        metric.add(loss * inputs.shape[0], sameNumber(outputs, labels), inputs.shape[0])
        test_loss = metric[0] / metric[2]
        test_acc = metric[1] / metric[2]
        testwriter.add_scalar('Loss', test_loss, i)
        testwriter.add_scalar('Accuracy', test_acc, i)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))
    return test_acc

if __name__ == "__main__":
    batch_size = 100
    lr = 0.001
    epochs = 10
    embedding_dim = 50
    vocab_size = 57080
    hidden_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainpath = r'data\train.txt'
    validatepath = r'data\validation.txt'
    testpath = r'data\test.txt'
    word2vec_pretrained = r'data\wiki_word2vec_50.bin'

    #step1: 生成训练、验证、测试数据
    word2id, id2word = build_word2id(trainpath, validatepath, testpath)
    print("length of word2id:",len(word2id))
    word2vecs = build_word2vec(word2vec_pretrained, word2id, save_to_path=None)
    print("length of word2vecs:", len(word2vecs))
    #训练数据
    traindata = CommentDataset(trainpath,word2id,id2word)
    trainloader = Data.DataLoader(traindata, batch_size, shuffle=True, num_workers=0, collate_fn=mycollate_fn)
    #验证数据
    validatedata = CommentDataset(validatepath, word2id, id2word)
    validateloader = Data.DataLoader(validatedata, batch_size, shuffle=True, num_workers=0, collate_fn=mycollate_fn)
    #测试数据
    testdata = CommentDataset(testpath, word2id, id2word)
    testloader = Data.DataLoader(testdata, batch_size, shuffle=False, num_workers=0, collate_fn=mycollate_fn)

    #step2: 建立模型
    lstmmodel = LSTMModel(embedding_dim, hidden_dim, pre_weight=word2vecs)
    textcnnmodel = TextCNN(vocab_size, embedding_dim, filters_num=128, filter_size=[1,3,5,7,9], pre_weight=word2vecs)

    #step3: 设置tensorboard可视化参数
    visdir = time.strftime('assets/visualize/' + lstmmodel.modelName + '_%m%d_%H_%M')
    trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    validatewriter = SummaryWriter('{}/{}'.format(visdir, 'Validate'))
    testwriter = SummaryWriter('{}/{}'.format(visdir, 'Test'))

    #step3: 训练模型
    trainedmodel = train(textcnnmodel, batch_size, lr, epochs, device, trainwriter, validatewriter, trainloader, validateloader)
    trainedmodel = train(lstmmodel, batch_size, lr, epochs, device, trainwriter, validatewriter, trainloader, validateloader)

    #step4: 测试模型
    testacc = test(trainedmodel, device, testloader, testwriter)