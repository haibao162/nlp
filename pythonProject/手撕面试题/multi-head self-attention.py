# 多头自注意力机制（Multi-Head Self-Attention）的核心思想是将输入数据分成多个“头”（heads），每个头学习不同的特征，然后将这些特征合并起来。其主要步骤包括：
# 线性变换：将输入数据分别通过三个线性变换，得到Query（Q）、Key（K）和Value（V）。
# 计算注意力分数：通过Q和K的点积计算注意力分数，并通过Softmax函数进行归一化。
# 加权求和：将注意力分数与V相乘，得到加权的特征表示。
# 多头合并：将多个头的输出合并，并通过一个线性变换输出最终结果。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.0):
        super(MultiHeadSelfAttention, self).__init__()
        """
        初始化多头自注意力机制
        :param embed_dim: 每个token的嵌入维度
        :param num_heads: 头的数量
        :param dropout: Dropout概率
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # assert self.head_dim * num_heads == embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入数据，形状为 (batch_size, seq_length, embed_dim)
        :param mask: 可选的掩码，形状为 (batch_size, seq_length)
        :return: 输出数据，形状为 (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length, embed_dim = x.size()

        # Q,K,V : batch_size * num_heads * seq_length * head_dim
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        # 提取权重
        attention = torch.nn.functional.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        # 加权求和
        out = torch.matmul(attention, V).transpose(1,2).contiguous().view(batch_size, seq_length, embed_dim)

        out = self.out(out)

        return out

if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embed_dim = 64
    num_heads = 4

    attention = MultiHeadSelfAttention(embed_dim, num_heads)

    x = torch.randn(batch_size, seq_length, embed_dim)

    output = attention(x)
    print(output.shape,'output')






                        


