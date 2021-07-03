#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： AutoPoetry
@File    ：main.py
@Author  ：郑家祥
@Date    ：2021/6/17 10:37 
@Description：实现模型训练、预测
'''
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from utils.PoetryData import poetryData
from models import PoetryModel
import time

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

def train(model,filename, batch_size, lr, epochs, device, trainwriter, pre_model_path=None):
    """
    描述：训练模型并计算损失
    """
    #step1: 模型初始化
    if pre_model_path:
        model.load(pre_model_path)
    model.to(device)

    #step2: 训练数据
    dataloader, ix2word, word2ix = poetryData(filename, batch_size)

    #step3: 定义目标函数与优化器，规定学习率衰减规则
    criterion  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    #step4: 开始训练
    metric = Accumulator(2)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            #data的shape为(batch_size, seq_len) --> 转置为(seq_len， batch_size)
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            """
            输入和预测的目标的对应关系应是如下所示:
                输入“床”的时候，网络预测的下一个字的目标是“前”。
                输入“前”的时候，网络预测的下一个字的目标是“明”。
                输入“明”的时候，网络预测的下一个字的目标是“月”。
                输入“月”的时候，网络预测的下一个字的目标是“光”。
                输入“光”的时候，网络预测的下一个字的目标是“,”。
            """
            input, target = data[:-1, :], data[1:, :]
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * data.shape[0], data.shape[0])
            train_loss = metric[0] / metric[1]
            trainwriter.add_scalar('Train Loss', train_loss, (epoch * len(dataloader)+ i))
            if i % 100 == 0:
                print('epoch: {:d}, batch: {:d}, Train Loss: {:.4f}'.format(epoch, i, train_loss))
        scheduler.step(train_loss)

    #step5: 迭代结束保存模型
    trainwriter.close()
    model.save()

def generate(model, filename, device, start_words, max_gen_len, prefix_words=None):
    """
    描述：给定开头诗句并允许指定诗词的意境
    """
    #step1: 设置模型参数
    _, ix2word, word2ix = poetryData(filename, 1)
    model.to(device)
    results = list(start_words)
    start_word_len = len(start_words)

    #step2: 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    #step3: 生成唐诗
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        # 读取第一句
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 生成后面的句子
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        # 结束标志
        if w == '<EOP>':
            del results[-1]
            break

    #step4: 返回结果
    return results

def generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic, prefix_words_acrostic):
    """
    描述：生成藏头诗
    """
    #step1: 设置模型参数
    _, ix2word, word2ix = poetryData(filename, 1)
    model.to(device)
    results = []
    start_word_len = len(start_words_acrostic)
    index = 0
    pre_word = '<START>'

    #step2: 设置第一个词为<START>
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    input = input.to(device)
    hidden = None

    #step3: 生成藏头诗
    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        if (pre_word in {u'。', u'！', '<START>'}):
            if index == start_word_len:
                break
            else:
                w = start_words_acrostic[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w

    # step4: 返回结果
    return results

if __name__ == "__main__":

    filename = r'data\tang.npz'
    batch_size = 16
    lr = 0.001
    epochs = 20
    vocab_size = 8293
    embedding_dim = 128
    hidden_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    #模型训练
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    visdir = time.strftime( 'assets/visualize/' + model.modelName + '_%m%d_%H_%M')
    trainwriter = SummaryWriter('{}/{}'.format(visdir, 'Train'))
    train(model, filename, batch_size, lr, epochs, device, trainwriter, pre_model_path=None)
    """

    """
    #给定开头生成诗句
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    model.load('models/PoetryModel_0617_18_39.pth')
    start_words = '飞流直下三千尺'
    max_gen_len = 128
    prefix_words = None
    poetry = ''
    result = generate(model, filename, device, start_words, max_gen_len, prefix_words)
    for word in result:
        poetry += word
        if word == '。' or word == '!':
            poetry += '\n'
    print(poetry)
    """

    #生成藏头诗
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
    model.load('models/PoetryModel_0617_18_39.pth')
    start_words_acrostic = '我爱中国'
    max_gen_len_acrostic = 128
    prefix_words_acrostic = None
    poetry = ''
    result = generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic, prefix_words_acrostic)
    for word in result:
        poetry += word
        if word == '。' or word == '!':
            poetry += '\n'
    print(poetry)



