import torch
import torch.nn as nn
import numpy as np

# 设置随机数种子
torch.manual_seed(0)


# 使用torch里的模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) # 3 * 5
    def forward(self, x):
        y_pred = self.layer1(x)
        return y_pred

#随机数
torch_model = TorchModel(3,5)

torch_x = torch.tensor([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])
print('torch预测的结果:', torch_model.forward(torch_x))
