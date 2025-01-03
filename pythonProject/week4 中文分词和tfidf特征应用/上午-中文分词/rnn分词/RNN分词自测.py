import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_run_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim, padding_idx=0)
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                num_layers=num_run_layers
                                )
        # rnn1 = nn.RNN(input_size=input_dim,hidden_size=hidden_size,batch_first=True)
        # rnn2 = nn.RNN(input_size=hidden_size,hidden_size=hidden_size,batch_first=True)
        self.classify = nn.Linear(hidden_size, 2) #二分类，因为输出的要么是0（不分词）要么是1（需要分词）。所以转成二维，然后对于y里的0和1而言，就表示选中这个二维的哪个权重
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        # print(x.shape, 'x.shape') 
        # 假设传入的形状是 3 * 20
        # torch.Size([3, 20])
        x = self.embedding(x)
        # embedding扩充用的input_dim，例如input_dim为5表示一个字符用5维向量表示
        # print(x.shape, 'embedding: x.shape') 
        # torch.Size([3, 20, 5]) embedding: x.shape
        x, _ = self.rnn_layer(x) # 3 * 20 * hidden_size ，重点是对每一行向量乘以矩阵 5(字符维度) * hidden_size
        y_pred = self.classify(x) #  (3 * 20 * hidden_size) * (hidden_size * 2) -> 3 * 20 * 2
        
        print(y.shape ,'y.shape')
        # torch.Size([3, 20]) y.shape
        print(y_pred.shape ,'y_pred')
        # torch.Size([3, 20, 2]) y_pred
        # print(y_pred.reshape(-1,2).shape ,'y_pred.reshape(-1,2)') # 60 * 2
        # print(y.view(-1).shape ,'y.view(-1)') # torch.Size([60]) y.view(-1)
        # print(y ,'y')
        # tensor([[   0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100],
        # [   1,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100]]) y
        
        if y is not None:
            #cross entropy
            #y_pred : n, class_num    [[1,2,3], [3,2,1]]   备注： 行数转化成(batch_size*sen_len, 2)
            #y      : n               [0,       1      ]

            #y:batch_size, sen_len  = 2 * 5
            #[[0,0,1,0,1],[0,1,0, -100, -100]]  y
            #[0,0,1,0,1,  0,1,0,-100.-100]    y.view(-1) shape= n = batch_size*sen_len
            # y_pred转化为x*2的矩阵，x自动计算，y转成一维数组，自动计算
            return self.loss_func(y_pred.reshape(-1,2), y.view(-1))
        else: 
            return y_pred

class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()
    def load(self):
        self.data = []
        with open(self.corpus_path, encoding='utf8') as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line) # 用结巴生成字符串的分词结果，我们去学习这个规律
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10:
                    break

    #将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence

#基于结巴生成分级结果的标注，即学习的规律用结巴生成结果y，我们去学习这个规律
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    # print(words, 'words')
    # ['同时', '国内', '有望', '出台', '新', '汽车', '刺激', '方案'] words
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    # print(label, 'label')
    # [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    return label

def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf8') as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab

#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length)
    # print(len(dataset), 'len(dataset)') # 假设是11，batch_size是3的话，会分成3+3+3+2一共四份。batch_size是2的话，那么生成的data_loader长度就是6
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #shuffle随机洗牌
    return data_loader

def main():
    epoch_num = 5 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    char_dim = 50 # 字的维度
    hidden_size = 100
    num_rnn_layers = 1 #rnn层数
    max_length = 20 #样本最大长度
    learning_rate = 1e-3
    vocab_path = 'chars.txt' #字表文件路径
    corpus_path = '../corpus.txt' #语料文件路径
    vocab = build_vocab(vocab_path) #建立字表
    data_loader = build_dataset(corpus_path, vocab,max_length,batch_size)
    model = TorchModel(char_dim, hidden_size,num_rnn_layers,vocab)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x,y in data_loader:
            optim.zero_grad()
            loss = model.forward(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 50
    hidden_size = 100
    num_rnn_layers = 1
    vocab = build_vocab(vocab_path)
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for input_string in input_strings:
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()

# sequence_to_label("同时国内有望出台新汽车刺激方案")

# 少量数据打印测试
epoch_num = 5 # 训练轮数
batch_size = 3 # 每次训练样本个数
char_dim = 5 # 字的维度
hidden_size = 3
num_rnn_layers = 1 #rnn层数
max_length = 20 #样本最大长度
learning_rate = 1e-3
vocab_path = 'chars.txt' #字表文件路径
corpus_path = '../corpus.txt' #语料文件路径
vocab = build_vocab(vocab_path) #建立字表
data_loader = build_dataset(corpus_path, vocab,max_length,batch_size)
model = TorchModel(char_dim, hidden_size,num_rnn_layers,vocab)
# print(len(data_loader), 'data_loader')
# 假设是数据总数是11个，batch_size是2的话，那么生成的data_loader就是6+5，一共两组。遍历data_loader会生成两次
for x,y in data_loader:
    loss = model.forward(x,y)

# if __name__ == "__main__":
    # main()
    # input_strings = ["同时国内有望出台新汽车刺激方案",
    #                  "沪胶后市有望延续强势",
    #                  "经过两个交易日的强势调整后",
    #                  "昨日上海天然橡胶期货价格再度大幅上扬"]
    
    # predict("model.pth", "chars.txt", input_strings)
