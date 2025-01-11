import torch
import torch.nn as nn
import numpy as np


x = np.array([[1, 2, 3],
              [3, 4, 5],
              [2, 4, 5],
              [1, 4, 5],
              [5, 6, 7]])

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=False,batch_first=True)
    def forward(self, x):
        return self.layer(x)
    
hidden_size = 4
torch_model = TorchRNN(3, hidden_size) # input_size必须等于x每一维的维度数量3
w_ih = torch_model.state_dict()['layer.weight_ih_l0']
w_hh = torch_model.state_dict()['layer.weight_hh_l0']
print(w_ih, w_ih.shape, 'w_ih') # 4 * 3
print(w_hh, w_hh.shape, 'w_hh') # 4 * 4   tanh(b + Wh + Ux)

torch_x = torch.FloatTensor([x])
output, h = torch_model.forward(torch_x)
# print(h)

print(output.detach().numpy(), 'torch模型预测结果')
# [[[-0.23121785 -0.7577772  -0.66730183 -0.95654255]
#   [-0.71892667 -0.8912347  -0.7666947  -0.9997174 ]
#   [-0.91781574 -0.9679548  -0.963919   -0.99999714]]]
print(h.detach().numpy(), 'torch模型预测隐含层结果') # 4* 4矩阵
print(h.detach().squeeze().numpy(), 'torch模型预测结果最后一个')
# [-0.05879186  0.9220749   0.9347278  -0.9998554 ]
print("---------------")

    
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size
    def forward(self, x):
        ht = np.zeros((self.hidden_size)) # (4,)矩阵
        # print(ht.shape, 'httt')
        output = []
        # 拿每一行去算，算好以后拼接进去
        for xt in x:
            ux = np.dot(self.w_ih, xt) # 4*3  (3,)  => (4,) (一维数组4个元素)。w_ih看做是U，和当前的输入x相乘
            # print(xt.shape, 'xtttt') # (3,)
            wh = np.dot(self.w_hh, ht) # 4*4  (4,)  => (4,) (一维数组4个元素)。将上一次的结果ht带进来，w_hh看做是权重W
            # print(ux, wh, 'uxxxxx') # [-0.2913686   1.56612682 -1.12666595  0.97105056] [0. 0. 0. 0.]
            ht_next = np.tanh(ux + wh)
            print(ht_next,ux,wh, 'ht_nextxtxtxt')
            # [-0.6206508   0.94479439 -0.42547424 -0.7515248 ]
            output.append(ht_next)
            ht = ht_next
        return np.array(output), ht
        
diy_model = DiyModel(w_ih, w_hh, hidden_size)
output, h = diy_model.forward(x)
print(output, "diy模型预测结果") # 5 * 4 ，原来的每一层3个变成了4个
print(h, "diy模型预测隐含层结果")
# 长度:batch_size * 向量维度: input_size  ->  长度: batch_size * rrn中的维度: hidden_size
# rnn是一行一行将上个结果带到下一行结果。

