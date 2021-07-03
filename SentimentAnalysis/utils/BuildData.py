#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：BuildData.py
@Author  ：郑家祥
@Date    ：2021/6/23 15:09 
@Description：生成word2id、id2word以及word2vec
'''
import torch
from zhconv import convert
import gensim
import numpy as np

def build_word2id(trainpath, validatepath, testpath):
    """
    :param file: word2id保存地址
    :return: 返回id2word、word2id
    """
    word2id = {'_PAD_': 0}
    id2word = {0: '_PAD_'}
    paths = [trainpath, validatepath, testpath]
    #print(path)
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                line = convert(line, 'zh-cn')
                words = line.strip().split()
                for word in words[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    for key, val in word2id.items():
        id2word[val] = key
    return word2id, id2word

def build_word2vec(file, word2id, save_to_path=None):
    """
    :param file: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    word2vecs = torch.from_numpy(word_vecs)
    return word2vecs


if __name__ == "__main__":
    trainpath = r'..\data\train.txt'
    validatepath = r'..\data\validation.txt'
    testpath = r'..\data\test.txt'
    word2vec_pretrained = r'..\data\wiki_word2vec_50.bin'
    word2id, id2word = build_word2id(trainpath, validatepath, testpath)
    print(len(word2id))


