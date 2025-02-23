import torch
import torch.nn as nn
import numpy as np

'''
用矩阵运算的方式复现一些基础的模型结构
清楚模型的计算细节，有助于加深对于模型的理解，以及模型转换等工作
'''

torch.manual_seed(0)
np.random.seed(42)

#构造一个输入
length = 6
input_dim = 12
hidden_size = 7
x = np.random.random((length, input_dim))
y = np.random.random((length, input_dim))

# print(x.shape) # 6 * 12

#使用pytorch的lstm层
torch_lstm = nn.LSTM(input_dim, hidden_size, batch_first = True) # 12 * 7
for key, weight in torch_lstm.state_dict().items():
    print(key, weight.shape)
# weight_ih_l0 torch.Size([28, 12]) # 12 * 7 * 4，计算的时候会转置，所以实际含义是12 * 28的矩阵
# weight_hh_l0 torch.Size([28, 7]) # 7 * 7 * 4
# bias_ih_l0 torch.Size([28])
# bias_hh_l0 torch.Size([28])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#将pytorch的lstm网络权重拿出来，用numpy通过矩阵运算实现lstm的计算
def numpy_lstm(x, state_dict):
    weight_ih = state_dict["weight_ih_l0"].numpy()
    weight_hh = state_dict["weight_hh_l0"].numpy()
    bias_ih = state_dict["bias_ih_l0"].numpy()
    bias_hh = state_dict["bias_hh_l0"].numpy()
    #pytorch将四个门的权重拼接存储，我们将它拆开
    w_i_x, w_f_x, w_c_x, w_o_x = weight_ih[0:hidden_size, :], \
                                 weight_ih[hidden_size:hidden_size*2, :],\
                                 weight_ih[hidden_size*2:hidden_size*3, :],\
                                 weight_ih[hidden_size*3:hidden_size*4, :]
    w_i_h, w_f_h, w_c_h, w_o_h = weight_hh[0:hidden_size, :], \
                                 weight_hh[hidden_size:hidden_size * 2, :], \
                                 weight_hh[hidden_size * 2:hidden_size * 3, :], \
                                 weight_hh[hidden_size * 3:hidden_size * 4, :]
    b_i_x, b_f_x, b_c_x, b_o_x = bias_ih[0:hidden_size], \
                                 bias_ih[hidden_size:hidden_size * 2], \
                                 bias_ih[hidden_size * 2:hidden_size * 3], \
                                 bias_ih[hidden_size * 3:hidden_size * 4]
    b_i_h, b_f_h, b_c_h, b_o_h = bias_hh[0:hidden_size], \
                                 bias_hh[hidden_size:hidden_size * 2], \
                                 bias_hh[hidden_size * 2:hidden_size * 3], \
                                 bias_hh[hidden_size * 3:hidden_size * 4]
    # Wi
    w_i = np.concatenate([w_i_h, w_i_x], axis=1)
    # Wf
    w_f = np.concatenate([w_f_h, w_f_x], axis=1)
    print(w_f.shape, 'w_f.shape')
    # (7, 19) w_f.shape, (7,12)和(7,7)合并
    w_c = np.concatenate([w_c_h, w_c_x], axis=1)
    w_o = np.concatenate([w_o_h, w_o_x], axis=1)
    b_f = b_f_h + b_f_x
    b_i = b_i_h + b_i_x
    b_c = b_c_h + b_c_x
    b_o = b_o_h + b_o_x
    print(b_f.shape, 'b_f.shape')
    # (7,) b_f.shape
    c_t = np.zeros((1, hidden_size))
    h_t = np.zeros((1, hidden_size))
    sequence_output = []
    for x_t in x:
        x_t = x_t[np.newaxis, :]
        print(x_t.shape, 'x_t.shape') # 1 * 12, 词向量
        hx = np.concatenate([h_t, x_t], axis=1)
        print(hx.shape, 'hx.shape') # (1, 19) hx.shape
        # f_t = sigmoid(np.dot(x_t, w_f_x.T) + b_f_x + np.dot(h_t, w_f_h.T) + b_f_h)
        f_t = sigmoid(np.dot(hx, w_f.T) + b_f) # 1 * 19 和 19 * 7合并
        i_t = sigmoid(np.dot(hx, w_i.T) + b_i)
        g = np.tanh(np.dot(hx, w_c.T) + b_c)
        c_t = f_t * c_t + i_t * g
        o_t = sigmoid(np.dot(hx, w_o.T) + b_o)
        h_t = o_t * np.tanh(c_t)
        sequence_output.append(h_t)
    return np.array(sequence_output), (h_t, c_t)


torch_sequence_output, (torch_h, torch_c) = torch_lstm(torch.Tensor([x]))
numpy_sequence_output, (numpy_h, numpy_c) = numpy_lstm(x, torch_lstm.state_dict())

print(torch.Tensor([x]).shape, 'torch.Tensor([x])')

# torch.Size([2, 6, 12]) , 输出变成torch.Size([2, 6, 7])。rnn复杂化
print(torch_sequence_output, torch_sequence_output.shape, 'torch_sequence_output')
# torch.Size([2, 6, 7]) torch_sequence_output

print(numpy_sequence_output, numpy_sequence_output.shape, 'numpy_sequence_output') # 传入的是 6 * 12矩阵
# (6, 1, 7) numpy_sequence_output
print("--------")                                               
print(torch_h.shape, 'torch_h.shape')
# torch.Size([1, 1, 7]) torch_h.shape
print(numpy_h.shape, 'numpy_h.shape') # 最后一个是1 * 7的                                                     
# (1, 7) numpy_h.shape
# print("--------")
# print(torch_c)
# print(numpy_c)

print(torch_h, 'torch_htorch_h')
# [[[-0.0137,  0.1415, -0.3028, -0.0158, -0.0789, -0.0658, -0.0485]]],
print(numpy_h, 'numpy_hnumpy_h')
# [[-0.01373187  0.14153321 -0.30284793 -0.01575391 -0.07888796 -0.06581849
#   -0.04854221]]


                        
                        
                        
    