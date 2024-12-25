import torch
import torch.nn as nn
import numpy as np

"""
基于pytorch的词向量CBOW
模型部分
"""

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, window_length):
        super(CBOW, self).__init__()
        self.word_vectors = nn.Embedding(vocab_size, embedding_size)
        self.pooling = nn.AvgPool1d(window_length)
        self.projection_layer = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, context):
        context_embedding = self.word_vectors(context)
        context_embedding = self.pooling(context_embedding.transpose(1,2)).squeeze()
        pred = self.projection_layer(context_embedding)
        return pred
    
vocab_size = 8 #词表大小 
embedding_size = 4 #人为指定的向量维度
window_length = 4 # 窗口长度
model = CBOW(vocab_size, embedding_size, window_length)
#假如选取一个词窗口【1,2,3,4,5】
context = torch.LongTensor([[1,2,4,5]]) #输入1,2,4,5, 预期输出3, 两边预测中间
pred = model(context)
print("预测值：", pred) # 8分类，我要预测是哪个值，那只能是词表里的任意字符都有可能。有多少词就有多少分类。
# 预测值： tensor([-1.1526,  0.5512,  1.0158,  0.3571, -0.8459,  0.0403,  0.3738,  0.4594],
#        grad_fn=<ViewBackward0>)