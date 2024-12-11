import torch
import numpy as np
import torch.nn as nn

x =torch.FloatTensor([1,2,3,4,5,6,7,8,9])
dp_layer = torch.nn.Dropout(0.5) #每个数都是50%概率保留，不是整体保留50%个数
dp_x = dp_layer(x) # 乘以1 / (1 - 0.5)
print(dp_x)