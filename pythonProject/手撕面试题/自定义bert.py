import torch
import math
import numpy as np
from transformers import BertModel
 
'''
通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models
'''
 
bert = BertModel.from_pretrained(r"/Users/mac/Documents/bert-base-chinese", return_dict=False)
# print(bert.state_dict())
# BertModel(
#   (embeddings): BertEmbeddings(
#     (word_embeddings): Embedding(21128, 768, padding_idx=0)
#     (position_embeddings): Embedding(512, 768)
#     (token_type_embeddings): Embedding(2, 768)
#     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#     (dropout): Dropout(p=0.1, inplace=False)
#   )
#   (encoder): BertEncoder(
#     (layer): ModuleList(
#       (0): BertLayer(
#         (attention): BertAttention(
#           (self): BertSdpaSelfAttention(
#             (query): Linear(in_features=768, out_features=768, bias=True)
#             (key): Linear(in_features=768, out_features=768, bias=True)
#             (value): Linear(in_features=768, out_features=768, bias=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (output): BertSelfOutput(
#             (dense): Linear(in_features=768, out_features=768, bias=True)
#             (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#         (intermediate): BertIntermediate(
#           (dense): Linear(in_features=768, out_features=3072, bias=True)
#           (intermediate_act_fn): GELUActivation()
#         )
#         (output): BertOutput(
#           (dense): Linear(in_features=3072, out_features=768, bias=True)
#           (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#   )
#   (pooler): BertPooler(
#     (dense): Linear(in_features=768, out_features=768, bias=True)
#     (activation): Tanh()
#   )
# )

state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110]) #假想成4个字的句子
torch_x = torch.LongTensor([x])
seqence_output, pooler_output = bert(torch_x)

print(seqence_output.shape, pooler_output.shape)
# torch.Size([1, 4, 768]) torch.Size([1, 768])
# print('所有权重', bert.state_dict().keys())  #查看所有的权值矩阵名称
# odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight',
#              'embeddings.token_type_embeddings.weight',
#                'embeddings.LayerNorm.weight', 
#                'embeddings.LayerNorm.bias', 
#                'encoder.layer.0.attention.self.query.weight', 
#                'encoder.layer.0.attention.self.query.bias', 
#                'encoder.layer.0.attention.self.key.weight', 
#                'encoder.layer.0.attention.self.key.bias', 
#                'encoder.layer.0.attention.self.value.weight', 
#                'encoder.layer.0.attention.self.value.bias', 
#                'encoder.layer.0.attention.output.dense.weight', 
#                'encoder.layer.0.attention.output.dense.bias', 
#                'encoder.layer.0.attention.output.LayerNorm.weight', 
#                'encoder.layer.0.attention.output.LayerNorm.bias', 
#                'encoder.layer.0.intermediate.dense.weight', 
#                'encoder.layer.0.intermediate.dense.bias', 
#                'encoder.layer.0.output.dense.weight', 
#                'encoder.layer.0.output.dense.bias', 
#                'encoder.layer.0.output.LayerNorm.weight', 
#                'encoder.layer.0.output.LayerNorm.bias', 
#                'pooler.dense.weight', 'pooler.dense.bias'])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict['embeddings.LayerNorm.weight'].numpy()
        self.embeddings_layer_norm_bias = state_dict['embeddings.LayerNorm.bias'].numpy()
        self.transformer_weights = []
        # transformer部分，多层，包括attention，mlp
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            # 前馈网络的处理
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
            self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
            self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x)
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))
        embedding = we + pe + te
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embedding
    
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x]) #不是从0开始的，是每个词对应的位置
    
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True))/ np.std(x, axis=1, keepdims=True) # (x - u) / 标准差
        x = x * w + b # 线性层
        return x
    
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x
    
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias, \
        output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b = weights

        attention_output = self.self_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                               self.num_attention_heads, self.hidden_size)
        
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)

        feed_forward_x = self.feed_forward(x, intermediate_weight, intermediate_bias, output_weight, output_bias)

        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x
    
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,num_attention_heads,hidden_size):
        q = np.dot(x, q_w.T) + q_b
        k = np.dot(x, k_w.T) + k_b
        v = np.dot(x, v_w.T) + v_b
        attention_head_size = int(hidden_size / num_attention_heads) # head_dim
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)

        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)

        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)

        qk = np.matmul(q, k.swapaxes(1, 2))
        qk = qk / np.sqrt(attention_head_size)

        qk = softmax(qk)
        qkv = np.matmul(qk,v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)

        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention


    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)
        return x
    
    def feed_forward(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x
    
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x
    
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output
    

db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)

torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)



                             
                           





                        