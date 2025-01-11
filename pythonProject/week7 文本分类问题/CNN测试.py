import torch
import torch.nn as nn

# 设置随机数种子
torch.manual_seed(0)

input = torch.randn(1, 1, 5)
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

weight = conv.state_dict()["weight"]
bias = conv.state_dict()["bias"]
print(weight, 'weight')
print(bias, 'bias') # bias长度等于out_channels通道数。对于每一个卷积核而言都是加一个常数
# tensor([[[-0.5516, -0.3824, -0.2380]]]) weight
# tensor([0.0214]) bias
print(input, 'input')
# tensor([[[ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845]]])

out = conv(input)
print(out, 'out')
# tensor([[[-0.1978,  0.8810,  1.2639]]]

a = torch.FloatTensor([[ 1.5410, -0.2934, -2.1788,]])
b = torch.FloatTensor([[-0.5516, -0.3824, -0.2380]])
bi = torch.FloatTensor([0.0214])
print(torch.sum(a * b) + 0.0214)


                        