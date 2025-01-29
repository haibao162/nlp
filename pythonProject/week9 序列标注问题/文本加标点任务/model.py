# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # BiLSTM: batch_size max_length input_dim 变成 batch_size  max_length hidden_size*2
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True,  bidirectional=True, num_layers=1)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss采用交叉熵损失

    def forward(self, x, target=None):
        #input shape:(batch_size, sen_len)
        x = self.embedding(x)
        #input shape:(batch_size, sen_len, input_dim)
        x, _ = self.layer(x) # batch_size, sen_len, hidden_size*2
        predict = self.classify(x) # batch_size, sen_len, class_num
        # target的形状为batch_size * sen_len，每个值的取值范围为class_num的类型值，也就是所有的分类
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                # mask false的不考虑计算
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict) # 结果是batch_size * sentence_len
            else:
                return predict # 如果没有用crf，返回的结构是batch_size, sen_len, class_num，最后一维代表分类的权重，每一类别都有权重，
                                # 值最大的位置就是预测的那个分类

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)