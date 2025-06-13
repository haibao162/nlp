import torch
from torchcrf import CRF
from torch.optim import Adam

# 设置随机数种子
torch.manual_seed(0)

# 参数
num_tags = 5 # 标签数量
seq_len = 3 # 系列长度
batch_size = 2 # 批量大小

# 创建CRF模型
model = CRF(num_tags, batch_first=True)

print(model.state_dict(), 'state_dict1')


# 模拟训练数据
trainset = torch.randn(batch_size, seq_len, num_tags) # 发射分数
traintags = (torch.rand([batch_size, seq_len]) * num_tags).floor().long() # 真实标签
print(trainset, trainset.shape, 'trainset') # torch.Size([2, 3, 5])
print(traintags, 'traintags') # batch_size, class_num = 2, 3
# tensor([[4, 4, 0],
#         [2, 2, 2]]) traintags

# 训练阶段
optimizer = Adam(model.parameters(), lr=0.05)
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    loss = - model(trainset, traintags) # CRF 损失函数为负对数似然
    print(f"Epoch {epoch}: Loss = {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

# 测试阶段
testset = torch.randn(batch_size, seq_len, num_tags)
model.eval()
decoded_tags = model.decode(testset)  # 用于预测序列的标签
print(decoded_tags) # 2 * 3
print(model.state_dict(), 'state_dict2')
# OrderedDict({'start_transitions': tensor([-1.6355, -1.5942,  1.8466, -1.8251,  1.1590]), 
# 'end_transitions': tensor([ 0.9988, -1.6395,  1.8473, -1.5772, -1.8538]), 
# 'transitions': tensor([[-1.9385, -1.7667, -2.4736, -1.7155, -1.7293],
#         [-1.7095, -1.5544, -2.0704, -1.5719, -1.6290],
#         [-2.0459, -1.9509,  1.9466, -1.5466, -2.2052],
#         [-1.5538, -1.5860, -2.2705, -1.5276, -1.7408],
#         [ 1.4102, -1.9733, -2.3985, -1.6790,  1.5056]])}) state_dict
