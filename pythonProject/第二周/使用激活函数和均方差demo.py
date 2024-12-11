import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1) #线性层
        self.activation = torch.sigmoid # sigmoid激活函数
        self.loss = nn.functional.mse_loss #均方差损失函数
    
    def forward(self, x, y = None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x) # (batch_size, 1) -> (batch_size, 1)
        # print('aaa', y)
        # tensor([[1.],
        # [1.]])
        print('bbb', y_pred)
        # bbb tensor([[0.4793],
        # [0.5021]], grad_fn=<SigmoidBackward0>)
        # return
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 第一个数大于第5个数，为正样本，否则为负样本
def build_sample():
    x = np.random.random(5)
    # [0.60060917 0.15887117 0.90274806 0.78877875 0.60275919]
    if (x[0] > x[4]):
        return x, 1
    else: return x, 0

def build_dataset(total_sample_num):
    X= []
    Y= []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    #X返回二维数组，Y就返回一维数组，输出类似于[0,1,0,1]的结果，表示X的每个样本的正负
    return torch.FloatTensor(X), torch.FloatTensor(Y)
    
# build_dataset(2)
# X: [array([0.74912369, 0.0352176 , 0.99632076, 0.69199515, 0.94953798]),
#  array([0.47137242, 0.65737101, 0.2043617 , 0.01214084, 0.35079724])]
# Y: [[0], [1]]

testX, testY = build_dataset(2)
testModel = TorchModel(5)
testModel.forward(testX, testY)
print('xxxxxxxx', testModel.forward(testX, testY))
# xxxxxxxx tensor(0.3115, grad_fn=<MseLossBackward0>)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # model.forward(x)
        for y_p, y_t in zip(y_pred, y): #预测标签，真实标签
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct+=1
            elif float(y_p) > 0.5 and int(y_t) == 1:
                correct+=1
            else:
                wrong+=1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num=20 #训练轮数
    batch_size=20 #每次训练样本个数
    train_sample=5000 # 每轮训练总共训练的样本总数
    input_size=5  # 输入向量维度
    learn_rate=0.001 #学习率
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr = learn_rate) #选择优化器
    log=[]
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x,y) # 计算loss  model.forward(x,y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("======第%d轮平均loss:%f"%(epoch+1, np.mean(watch_loss)) )
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model2.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# main()











