import torch
import torch.nn as nn

# 设置随机数种子
torch.manual_seed(0)

# Conv2d
# in_channels: 输入通道数
# out_channels: 输出通道数
# kernel_size: 卷积核大小

# (batch_size, in_channels, height, width)
input_tensor = torch.randn(1,3,32,32)

# 广泛应用于图像处理
# 定义卷积层
conv_layer = nn.Conv2d(
    in_channels=3,  # 输入通道数
    out_channels=16,  # 输出通道数
    kernel_size=3,  # 卷积核大小
    stride=1,  # 步长
    padding=1,  # 填充大小
    dilation=1,  # 膨胀率
    groups=1,  # 分组数
    bias=True,  # 是否添加偏置
    padding_mode='zeros'  # 填充模式
)

# 应用卷积操作
output_tensor = conv_layer(input_tensor)
print("输出张量的形状:", output_tensor.shape)  # 输出形状：(1, 16, 32, 32)


                        