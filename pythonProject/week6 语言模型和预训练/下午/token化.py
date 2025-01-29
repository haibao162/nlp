import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer


'''
关于transformers自带的序列化工具
模型文件下载 https://huggingface.co/models
'''

tokenizer = BertTokenizer.from_pretrained(r"/Users/mac/Documents/bert-base-chinese")
print(tokenizer)
string = "咱呀么老百姓今儿个真高兴"
#分字
tokens = tokenizer.tokenize(string)
print("分字：",tokens)

#编码，前后自动添加了[cls]和[sep]
encoding = tokenizer.encode(string)
print("编码：", encoding)

string1 = "今天天气真不错"
string2 = "明天天气怎么样"
encoding = tokenizer.encode(string1, string2)
print("文本对编码：", encoding)
#同时输出attention_mask和token_type编码
encoding = tokenizer.encode_plus(string1, string2)
print("全部编码：", encoding)
# 全部编码： {'input_ids': 
#        [101, 791, 1921, 1921, 3698, 4696, 679, 7231, 102, 3209, 1921, 1921, 3698, 2582, 720, 3416, 102], 
#        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

# 0 作为padding
input_id = tokenizer.encode(string1, max_length=30, pad_to_max_length=True)
print('--------')
print(input_id, 'input_id')

encode = tokenizer.encode_plus(string1, max_length=30, pad_to_max_length=True)
print(encode, 'encodeeeee')

