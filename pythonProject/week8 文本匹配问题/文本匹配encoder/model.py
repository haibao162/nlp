# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        # torch.Size([128, 20]) torch.Size([128, 1]) target
        x = self.embedding(x)
        x = self.layer(x)
        x = self.layer2(x)
        print(x.shape)
        # torch.Size([2, 4, 128])

        #pooling操作默认对于输入张量的最后一维进行，这里用的max_pool1d
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze() # 句向量做pooling
        # print(x.shape)
        # torch.Size([2, 128])
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # margin: float = 0.,
        # y = 1, loss = 1 - cos(a,b)。cos(a,b)训练成1。
        # y = -1, loss = max(0, cos(a,b) - margin)。margin默认是0，如果cos已经小于0，就认为不相似，loss=0不训练
        # cos(a,b)大于0，loss= cos(a,b) - margin = cos(a,b)。loss往0训练，意味着cos(a,b)的值训练成0，变成正交的。
        self.loss = nn.CosineEmbeddingLoss()
    
    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1) # 计算余弦值
        return 1 - cosine
    
    # 三元组
    # a为原点，p为正样本，n为负样本。margin为边距
    # loss = max(d(a,p) - d(a, n) + margin, 0)
    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p) # tensor(0.1)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        # tensor(True) 
        return torch.mean(diff[diff.gt(0)]) #greater than
    
    #sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, target=None):
        #同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            #如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            #如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)
        
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    
if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[-1]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())
                        