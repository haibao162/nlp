import torch
import torch.nn as nn

#pooling操作默认对于输入张量的最后一维进行
#入参5，代表把五维池化为一维。这里是4。对文本的4个字符池化（这里是求平均），而不是一个文本的5个向量
layer = nn.AvgPool1d(4)
# 3条文本，文本长度是4，向量长度是5
x = torch.rand(3,4,5)
# print(x.shape)
x = x.transpose(1,2) #交换第二和第三维
# print(x.shape)
# torch.Size([3, 5, 4])
print(x)
y = layer(x)
print(y)
print(y.shape, 'y.shape') # torch.Size([3, 5, 1]) 
y = y.squeeze()
print(y)
print(y.shape) # 3 * 5

