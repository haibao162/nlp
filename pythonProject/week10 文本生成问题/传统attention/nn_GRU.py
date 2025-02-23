import torch
import torch.nn as nn

torch.manual_seed(0)

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        print(x.shape, 'init size')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU前向传播
        out, hidden = self.gru(x, h0)

        print(out.shape, 'out.shape') # torch.Size([2, 5, 3]) out.shape
        print(out, 'out all') # 
        print(out[0,0].shape, 'out[0,0]') # torch.Size([3])

        print(hidden.shape, 'hidden.shape') # torch.Size([1, 2, 3])  num_layers * 2 * 3
        out = out[:,-1,:]
        # 获取最后一个
        print(out.shape, 'out.shape2') # torch.Size([2, 3]) out.shape2

        print(hidden, 'hidden value')
        # [[[ 0.1983, -0.6420, -0.3143],
        #  [-0.9219,  0.2455, -0.1549]]]
        print(out, 'out value')
        # [[ 0.1983, -0.6420, -0.3143],
        # [-0.9219,  0.2455, -0.1549]]


        # 取最后一个时间步的输出
        out = self.fc(out)
        return out
        
# 定义模型参数
input_size = 20     # 每个时间步的特征数
hidden_size = 3    # 隐状态的维度
output_size = 1     # 输出的维度
num_layers = 1      # GRU层数

model = GRUNet(input_size, hidden_size, output_size, num_layers)

x = torch.randn(2,5,input_size)
res = model(x)
print(res.shape) # torch.Size([2, 1])
# print(x)

# print(model)