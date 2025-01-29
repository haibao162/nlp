import torch
import torch.nn as nn

# 实例化一个 LSTM 对象
input_size = 10 # 句向量维度
hidden_size = 20 
num_layers = 2

lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

# 创建一个输入张量 3 * 5 * 10
batch_size = 3
sequence_length = 5
input = torch.randn(batch_size, sequence_length, input_size)

ouput, (hc, tc) = lstm(input)
print(input.shape, 'input') # 3  5  10

print(ouput.shape, 'output') # 3 * 5 * 20

print(hc.shape, 'hc.shape') # torch.Size([2, 5, 20]) hc.shape， 如果是双层的话

# BiLSTM
lstm2 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True)
ouput2, (hc2, tc2) = lstm2(input)
print(ouput2.shape, 'ouput2.shape') #torch.Size([3, 5, 40]) ouput2.shape
print(hc2.shape, 'hc2.shape') #torch.Size([4, 5, 20]) hc2.shape






 
