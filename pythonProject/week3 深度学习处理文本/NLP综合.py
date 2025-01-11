#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""
class TorchModel(nn.Module):
    # vector_dim表示几个数表示一个字符,即字符的向量维度。sentence_length是句子长度
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.pool = nn.AvgPool1d(sentence_length) #池化层   从每个字符的向量里各取一部分，sentence_length行凑成一行向量，表示当前字符串。
        # 之所以可以这样是因为ab和bc的权重不一样，各取一部分，他们的差就是a和c权重的差，是固定的，所以可以加起来池化的值可以认为能唯一确定字符串。
        self.classify = nn.Linear(vector_dim, 1) # 单个任务，转成一维
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss # 一维的输入，一维的结果，一起用均方差作为loss函数。
    def forward(self, x, y = None):
        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        
# 为每个字符标号
def build_vocab():
    chars = '你我他abcdefghijklmnopqrstuvwxyz'
    vocab = { "pad": 0}
    for index,char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set('你我他') & set(x):
        y = 1
    else: 
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x,y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x,y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evalute(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) > 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print('正确预测个数：%d,正确率：%f'%(correct, correct/(correct + wrong)))
    return correct/(correct + wrong)

def main():
    epoch_num = 10 #训练轮数
    batch_size = 20 #每次训练样本个数
    train_sample = 500 #每轮训练总共训练的样本总数
    char_dim = 20 #每个字的维度
    sentence_length = 6 # 样本文本长度
    learning_rate = 0.005 # 学习率
    #建立字表
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evalute(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model3.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
                        
def predict(model_path, vocab_path, input_strings):
    print('predictpredictpredictpredictpredict')
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    model = build_model(vocab, char_dim,sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print('输入：%s,预测类别：%d，概率值：%f' %(input_string, round(float(result[i])), result[i]))

main()
test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
predict("model3.pth", "vocab.json", test_strings)