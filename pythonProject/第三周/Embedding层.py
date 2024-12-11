import torch
import torch.nn as nn

num_embeddings = 7 #字符集字符总数
embedding_dim = 6 #字符用6个向量表示
embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
print('随机初始化权重')
print(embedding_layer.weight, 'embedding_layer.weight') # 7 * 6
print('###########')

#字符集
vocab = {
    '[pad]': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6
}

def str_to_sequence(string, vocab):
    return [vocab[s] for s in string]

#如果长度不一样，可以去掉少数太长的，长度不够的可以补0
string1 = 'abcde'
string2 = 'ddccb'
string3 = 'fedab'

sequence1 = str_to_sequence(string1, vocab)
sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)
# print(sequence3) #[6, 5, 4, 1, 2]

x = torch.LongTensor([sequence1,sequence2, sequence3])
# print(x, 'x') # 3 * 5
embedding_out = embedding_layer(x)
print(embedding_out)
print(embedding_out.shape) # 3 * 5 * 6
print(embedding_out.detach())


