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

# 模拟训练数据
trainset = torch.randn(batch_size, seq_len, num_tags) # 发射分数
traintags = (torch.rand([batch_size, seq_len]) * num_tags).floor().long() # 真实标签
print(trainset, trainset.shape, 'trainset') # torch.Size([2, 3, 5])
print(traintags, 'traintags')
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
