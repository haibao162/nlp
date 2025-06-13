import torch
import torch.nn.functional as F
import torch as nn
import math

torch.manual_seed(1)

x = torch.randn(2,3,4)
print(x)
print(x.size())
print(x.transpose(1, 2))
a, b, c = x.size()
print(a,b,c, 'xxx')

a = 61
b = 4
print(a // b)

        
