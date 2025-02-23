import torch
from torch import nn

# 假设有一个简单的模型输出和标签
y_pred = torch.randn(64, 10)  # 假设有64个样本，每个样本有10个分类的输出
y_label = torch.randint(0, 10, (64,))  # 随机生成标签索引

loss_func = nn.CrossEntropyLoss()
loss = loss_func(y_pred, y_label)
print(y_pred.shape, y_label.shape)
a = torch.tensor([[1,2]])
print(torch.sigmoid(a))
