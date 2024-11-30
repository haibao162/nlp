import torch
import torch.nn as nn
import numpy as np

# 使用torch里的模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) # 3 * 5
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 5 * 2
    def forward(self, x):
        x = self.layer1(x)
        y_pred = self.layer2(x)
        return y_pred

#随机数
torch_model = TorchModel(3,5,2)

torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy() #打印的是5*3
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
print('state_dict', type(torch_model.state_dict()))
print(torch_model_w1, "torch w1 权重")
# print(torch_model_b1, "torch b1 权重")
print("-----------")
print(torch_model_w2, "torch w2 权重")
# print(torch_model_b2, "torch b2 权重")

torch_x = torch.tensor([[3.1, 1.3, 1.2],
              [2.1, 1.3, 13]])
print('torch预测的结果:', torch_model.forward(torch_x))

class DiyModel:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
    def forward(self, x):
        # x为2*3的，乘以3*5，在乘以5*2
        hidden = np.dot(x, self.w1.T) + self.b1
        y_pred = np.dot(hidden, self.w2.T) + self.b2
        return y_pred
    
diyModel = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
print('自定义预测的结果:', diyModel.forward(torch_x.numpy()))
