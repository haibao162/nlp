import torch
import torch.nn as nn


# 1. RNN 送入单个数据
def test01():

    # 输入数据维度 128, 输出维度 256
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 第一个数字: 表示句子长度
    # 第二个数字: 批量个数
    # 第三个数字: 表示数据维度
    inputs = torch.randn(1, 1, 128)
    hn = torch.zeros(1, 1, 256)

    output, hn = rnn(inputs, hn)
    print(output.shape)
    print(hn.shape)


# 2. RNN层送入批量数据
def test02():

    # 输入数据维度 128, 输出维度 256
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 第一个数字: 表示句子长度
    # 第二个数字: 批量个数
    # 第三个数字: 表示数据维度
    inputs = torch.randn(1, 32, 128)
    hn = torch.zeros(1, 32, 256)

    output, hn = rnn(inputs, hn)
    print(output.shape)
    print(hn.shape)


if __name__ == '__main__':
    test01()
    test02()