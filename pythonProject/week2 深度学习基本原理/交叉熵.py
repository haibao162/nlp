import torch
import torch.nn as nn
import numpy as np

ce_loss = nn.CrossEntropyLoss()

pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]])
target= torch.LongTensor([1,2,0])

pred2 = torch.Tensor([[0.5, 0.4, 0.1, 0.3]])
target2 = torch.tensor([0])

loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")
# print(ce_loss.state_dict().keys(), 'state_dict')



def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis = 1, keepdims = True)

# print(softmax(pred.numpy()))
# print(torch.softmax(pred, dim=1))

def to_one_hot(target,shape):
    one_hot_target = np.zeros(shape)
    for i,t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target
#手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis = 1)
    print(pred, target, sum(entropy) / batch_size, 'shuchu')
    return entropy 

# print(to_one_hot(target, pred.shape))
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
cross_entropy(pred.numpy(), target.numpy())
# print(pred.shape, pred.shape[0])

loss2 = ce_loss(pred2, target2)
print(loss2, 'loss2loss2')
# tensor(0.9459)
cross_entropy(pred2.numpy(), target2.numpy())
