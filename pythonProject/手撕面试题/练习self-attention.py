import torch
import torch.nn.functional as F
import torch.nn as nn

import math

torch.manual_seed(1)

# x = torch.randn(2,3,4)
# print(x)
# print(x.transpose(1, 2))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim # hidden_size
        self.num_heads = num_heads # 分成多少头
        self.head_dim = embed_dim // num_heads # 每一头多少维度

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x, mask=None, y = None):
        batch_size, seq_length, embed_dim = x.size()

        # Q,K,V : batch_size * num_heads * seq_length * head_dim
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)


        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        # attention = F.softmax(attention, dim=-1)

        attention = torch.nn.functional.softmax(scores, dim=-1)

        attention = self.dropout(attention)


        output = torch.matmul(attention, V).transpose(1,2).contiguous().view(batch_size, seq_length, self.embed_dim)

        output = self.out(output)

        return output



if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embed_dim = 64
    num_heads = 4

    attention = MultiHeadSelfAttention(embed_dim, num_heads)

    x = torch.randn(batch_size, seq_length, embed_dim)

    output = attention(x)
    print(output.shape,'output')
    print(output,'output22')




    