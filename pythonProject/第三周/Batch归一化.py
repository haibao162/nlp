import torch
import numpy as np

x = np.random.random((4,5))

bn = torch.nn.BatchNorm1d(5)
y = bn(torch.FloatTensor(x))

print(x, 'random x')
# [[0.97870085 0.35138781 0.19070527 0.54786248 0.0973376 ]
#  [0.3590548  0.85239535 0.72844752 0.77597165 0.55513752]
#  [0.25496997 0.76611472 0.94096325 0.25996571 0.67144259]
#  [0.62800166 0.77209578 0.73736095 0.85093966 0.16904443]]
print(y, 'torch y')
# tensor([[-1.4523, -1.5528,  0.5608, -0.0463, -0.7000],
#         [ 0.0578,  0.9234, -1.7292,  0.9484,  1.1162],
#         [ 1.3736, -0.2036,  0.5060,  0.7089, -1.2514],
#         [ 0.0210,  0.8330,  0.6624, -1.6109,  0.8352]],
#        grad_fn=<NativeBatchNormBackward0>) torch y
# print(bn.state_dict())
# OrderedDict({'weight': tensor([1., 1., 1., 1., 1.]), 
#              'bias': tensor([0., 0., 0., 0., 0.]), 
#              'running_mean': tensor([0.0281, 0.0354, 0.0366, 0.0385, 0.0536]),
#                'running_var': tensor([0.9027, 0.9018, 0.9100, 0.9055, 0.9077]), 
#                'num_batches_tracked': tensor(1)})

gamma = bn.state_dict()["weight"].numpy()
beta = bn.state_dict()["bias"].numpy()
num_features = 5
eps = 1e-05
momentum = 0.1

running_mean = np.zeros(num_features)
running_var = np.zeros(num_features)

mean = np.mean(x, axis=0)
var = np.var(x, axis=0) # 方差 np.var([1,2,3,4])的值为1.25

# print(running_mean, 'running_mean')

running_mean = momentum * running_mean + (1 - momentum) * mean
running_var = momentum * running_var + (1 - momentum) * var

x_norm = (x - mean) / np.sqrt(var + eps) # (x - 均值) / sqrt（方差 + ξ）

y = gamma * x_norm + beta #  y = BN(x)= gamma * x + beta, gamma和beta为求的权重
print(x_norm, 'x_norm')
print(y, 'ours y')
# print(gamma, 'gamma')
# [1. 1. 1. 1. 1.]
# print(running_mean)
# [0.49966364 0.61694857 0.58443232 0.54781639 0.33591648]
# print(running_var)
# [0.07048441 0.03453358 0.06961401 0.04769475 0.05396163]
