# 尝试修改nlpdemo，做一个6分类任务，判断特定字符a在字符串的第几个位置，使用rnn和交叉熵。
#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import copy
import json
import matplotlib.pyplot as plt

# 判断字符a的位置
special_str = 'a'

# vector_dim = 20 #每个字的维度
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # padding_idx补齐长度
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0) # 文本个数 * 文本长度6 * vector_dim
        # self.pool = nn.AvgPool1d(sentence_length) #池化层  文本个数 * vector_dim
        # self.classify = nn.Linear(vector_dim, 6) # 
        self.layer = nn.RNN(vector_dim, vector_dim, bias=False,batch_first=True)
        self.classify = nn.Linear(vector_dim, 6) #a一定会有，随机插入到文本的某个位置
        # self.activation = torch.sigmoid
        # self.loss = nn.functional.mse_loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.embedding(x)
        print(x, x.shape, 'embedding层')
        x, h = self.layer(x)
        print(x, x.shape, 'RNN层')
        x = x[:, -1, :]
        print(x, x.shape, '输出层')
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        
# 为每个字符标号
def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = { "pad": 0}
    for index,char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机选取sentence_length个字
    newVocab = copy.deepcopy(vocab)
    newVocab.pop(special_str) # 不会随机生成字符a
    x = [random.choice(list(newVocab.keys())) for _ in range(sentence_length)]
    y = random.randint(0, 5) #随机生成字符a的位置
    x[y] = special_str #位置上的字符赋值为a
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x,y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x,y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


vocab = build_vocab()
testx, testy = build_dataset(2, vocab, 6)
print(testx, 'testx')
print(testy, 'testy')
testModel = TorchModel(20, 6, vocab)
testModel.forward(testx, testy)
print('xxxxxxxx', testModel.forward(testx, testy))
