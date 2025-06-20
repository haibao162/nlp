# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.classify = nn.Linear(hidden_size, class_num)
        self.pool = nn.AvgPool1d(max_length) # pooling将句子每个词的词向量加起来再求平均，用一个向量来表示这个句子
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x) #input shape:(batch_size, sen_len)
        x = self.layer(x)  #input shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)
        x = self.pool(x.transpose(1,2)).squeeze() # 池化以后：batch_size * input_dim，每个句子用一个词向量表示
        # rnn不需要池化，最后一个结果就是输出
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict
        
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    # Config["class_num"] = 10
    model = TorchModel(Config)