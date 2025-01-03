import  torch
import torch.nn as nn
import numpy as np

"""
2003年的论文
基于pytorch的语言模型
与基于窗口的词向量训练本质上非常接近
只是输入输出的预期不同
不使用向量的加和平均，而是直接拼接起来
"""

"""
假设语料库有5000个词，每个词用128维进行表示，嵌入的矩阵就是5000 * 128
每个词对应一个5000维的one-hot向量，那么一个有4个词的一句话可以用4*5000的one-hot矩阵表示
2 * 4 -> 2 * 4 * 5000 -> 2 * 4 * 128
将one-hot矩阵和embedding矩阵（嵌入的矩阵）相乘，得到4*128维矩阵
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.word_vectors = nn.Embedding(vocab_size, embedding_size)
        self.inner_projection_layer = nn.Linear(embedding_size * max_len, hidden_size)
        self.outter_projection_layer = nn.Linear(hidden_size, hidden_size)
        self.x_projection_layer = nn.Linear(embedding_size * max_len, hidden_size)
        self.projection_layer = nn.Linear(hidden_size, vocab_size)

    #根据论文y = Wx + Utanh(hx+d) + b
    def forward(self, context):
        print(context.shape, 'x.shape') # 2 * 4
        #context shape = batch_size, input_size(max_len)   5条数据，每条数据用10个字符表示
        context_embedding = self.word_vectors(context) # output shape: batch_size * input_size(max_len) * vector_dim(embedding_size)
        print(vocab_size,embedding_size, context_embedding.shape, 'context_embedding.shape') # 8 5 torch.Size([2, 4, 5])
        ### 上面要注意embedding层，先将2 * 4 变成2 * 4 * 词表大小，在乘以 (词表大小 * embedding_size)，变成2 * 4 * 5,5表示我们这里用5个向量表示一个字符
        #总体计算 y = b+Wx+Utanh(d+Hx)， 其中x为每个词向量的拼接
        #词向量的拼接
        x = context_embedding.view(context_embedding.shape[0], -1) #shape = batch_size, max_length*embedding_size
        #hx + d
        inner_projection = self.inner_projection_layer(x) #shape = batch_size, hidden_size(embedding_size)
        #tanh(hx+d)
        inner_projection = torch.tanh(inner_projection)  #shape = batch_size, hidden_size
        #U * tanh(hx+d) + b
        outter_project = self.outter_projection_layer(inner_projection) # shape = batch_size, hidden_size
        #Wx
        x_protection = self.x_projection_layer(x)
        #y = Wx + Utanh(hx+d) + b
        y = x_protection + outter_project
        #softmax后输出预测概率, 训练的目标是让y_pred对应到字表中某个字
        y_pred = torch.softmax(y, dim=-1)  #shape = batch_size, hidden_size
        return y_pred


vocab_size = 8 # 词表大小
embedding_size = 5 #人为指定的向量维度
max_len = 4 # 输入长度
hidden_size = vocab_size #由于最终的输出维度应当是字表大小的，所以这里hidden_size = vocab_size
model = LanguageModel(vocab_size, max_len, embedding_size, hidden_size)

#假如选取一个文本窗口“天王盖地虎”
#输入：“天王盖地” —> 输出："虎"
#假设词表embedding中, 天王盖地虎 对应位置 12345
context = torch.LongTensor([[1,2,3,4],[2,3,4,5]])
pred = model(context)
print("预测值：", pred) # 8分类任务，需要训练其中的参数
# tensor([[0.1484, 0.0501, 0.1216, 0.1622, 0.1084, 0.0842, 0.1422, 0.1830],
#         [0.1387, 0.1582, 0.1352, 0.1414, 0.0297, 0.1800, 0.0545, 0.1623]],
#        grad_fn=<SoftmaxBackward0>)
print("loss使用交叉熵", nn.functional.cross_entropy(pred, torch.LongTensor([5, 1])))


print("词向量矩阵")
matrix = model.state_dict()["word_vectors.weight"]
# torch.Size([8, 5])
 
print(matrix.shape)  #vocab_size, embedding_size
print(matrix)