import torch
from torch import nn



mse_loss = nn.functional.mse_loss

a = torch.FloatTensor([[1], [2]])
b = torch.FloatTensor([[2], [4]])
print(mse_loss(a, b))
# 2.5
