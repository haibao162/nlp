import torch
import numpy as np

print(np.exp(1))

x = [1,2,-2,-1]
print(torch.tensor(x))
print(torch.Tensor(x))

print(torch.softmax(torch.Tensor(x), dim=0), 'torch实现的softmax')


def softmax(p):
    res = []
    for i in p:
        res.append(np.exp(i))
    total = sum(res)
    other = [(i / total) for i in res]
    return other

print(softmax(x), '自己实现的softmax')