import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__ (self, embed_dim, head_nums):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dims = embed_dim // head_nums
        self.head_nums = head_nums # 多少头
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None, y = None):
        batch_size, seq_length, hidden_size = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.head_nums, self.head_dims).transpose(1,2)
        K = K.view(batch_size, seq_length, self.head_nums, self.head_dims).transpose(1,2)
        V = V.view(batch_size, seq_length, self.head_nums, self.head_dims).transpose(1,2)


        scores = torch.matmul(Q,K.transpose(-2, -1)) / math.sqrt(self.head_dims)

        if mask is not None:
            scores = torch.masked_fill(mask==0,float('-inf'))

        scores = F.softmax(scores, dim=-1)

        attention = self.dropout(scores)

        output = torch.matmul(attention, V).transpose(1,2).contiguous().view(batch_size, seq_length,self.embed_dim)

        output = self.out(output)

        return output

if __name__ == "__main__":
    batch_size = 2
    head_nums = 4
    seq_length = 10
    embed_dim = 64

    x = torch.randn(batch_size, seq_length, embed_dim)
    model = MultiHeadSelfAttention(embed_dim, head_nums)
    attention = model(x)
    print(attention.shape, 'xx')