import torch
import torch.nn as nn

# 设置随机数种子
torch.manual_seed(1)

#bce loss pytorch使用
sig = nn.Sigmoid()
input = torch.randn(5) #随机构造一个模型当前的预测结果 y_pred
print(input, 'input')
# tensor([ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519])
input = sig(input)
print(input, 'sig input')
# tensor([0.6596, 0.5663, 0.5154, 0.6505, 0.3889]) sig input
target = torch.FloatTensor([1,0,1,0,1]) # y_pred
bceloss = nn.BCELoss() # bce loss
loss = bceloss(input, target)
print(loss) #tensor(0.7820)

l = 0
for x,y in zip(input, target):
    l += y * torch.log(x) + (1-y) * torch.log(1 - x)

l = -l / 5
print(l) #tensor(0.7820)

# 电商评论分类：好评/差评
# 训练集/验证集划分   
# 数据分析：正负样本数，文本平均长度等
# 实验对比3种以上模型结构的分类效果
# 每种模型对比模型预测速度
# 总结成表格输出
