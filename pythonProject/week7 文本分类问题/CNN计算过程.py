
import torch
import torch.nn as nn
import numpy as np



#使用pytorch的1维卷积层

input_dim = 6
hidden_size = 8
kernel_size = 2
#  in_channels: int,out_channels: int, kernel_size: _size_1_t, 卷积核大小是一维的
torch_cnn1d = nn.Conv1d(input_dim, hidden_size, kernel_size)
# weight torch.Size([8, 6, 2]) # 卷积核是6 * 2的

for key, weight in torch_cnn1d.state_dict().items():
    print(key, weight.shape)

x = torch.rand((6, 8))  #embedding_size * max_length
# print(x.shape)

def numpy_cnn1d(x, state_dict):
    weight = state_dict["weight"].numpy()
    bias = state_dict["bias"].numpy()
    sequence_output = []
    for i in range(0, x.shape[1] - kernel_size + 1):
        window = x[:, i:i + kernel_size]
        kernel_outputs = []
        for kernel in weight: 
            kernel_outputs.append(np.sum(kernel * window))
        sequence_output.append(np.array(kernel_outputs) + bias)
    return np.array(sequence_output).T

print(x.shape, 'x.shape')
# print(torch_cnn1d(x.unsqueeze(0)), '')
print(torch_cnn1d(x.unsqueeze(0)), 'torch_cnn1d') # 8 * 7
print(numpy_cnn1d(x.numpy(), torch_cnn1d.state_dict()), '自己计算的')

                        