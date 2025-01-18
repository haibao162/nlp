import torch
import torch.nn as nn

# reduction设为none便于查看损失计算的结果
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = triplet_loss(anchor, positive, negative)
print(output, 'output')
output.backward()

# output = triplet_loss(anchor, positive, negative)
# print(output, 'output')





                        