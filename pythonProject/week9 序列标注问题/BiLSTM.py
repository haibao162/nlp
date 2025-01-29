import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义双向LSTM
        # hidden_size=20
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        print(h0.shape, x.shape, 'h000')
        # torch.Size([4, 32, 20]) torch.Size([32, 5, 10]) h000

        # 通过 LSTM
        # h0初始隐藏状态，c0初始记忆单元。hc表示输出隐藏状态和最终的记忆单元。
        out, _ = self.lstm(x, (h0, c0)) # out 的形状为 (batch_size, seq_length, hidden_size * 2)
        print(out.shape, 'out.shape')
        # torch.Size([32, 5, 40])
        out = out[:, -1, :]
        print(out.shape, 'out.shape22')
        # torch.Size([32, 40])

        # 通过全连接层
        out = self.fc(out) # 取最后一个时间步的输出
        return out

# 示例：假设输入特征维度为 10，LSTM 隐层大小为 20，2 层 LSTM，输出大小为 1
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1

model = BiLSTM(input_size, hidden_size, num_layers, output_size)

# 假设输入的 batch 大小为 32，序列长度为 5，特征维度为 10
inputs = torch.randn(32, 5, input_size)

# 前向传播
outputs = model(inputs)
print(outputs.shape)  # 输出大小为 (32, 1)


# Examples::
#         >>> rnn = nn.LSTM(10, 20, 2)
#         >>> input = torch.randn(5, 3, 10)
#         >>> h0 = torch.randn(2, 3, 20)
#         >>> c0 = torch.randn(2, 3, 20)
#         >>> output, (hn, cn) = rnn(input, (h0, c0))


#  准确率和召回率：一个句子三个实体，预测了一个且是正确的，召回率是33.3%，准确率是100%
# F1: 准确和召回的平均
# Marco-F1计算每个类别的F1，所有的F1取平均
# Micro-F1所有类别样本合并计算准确和召回，在计算F1
# 区别在于考虑样本数量的均衡，如果Marco-F1和Micro-F1差别比较大，某一类的数量比较多，样本不均衡
