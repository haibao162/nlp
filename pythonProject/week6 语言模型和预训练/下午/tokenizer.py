import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
'''

关于transformers自带的序列化工具
模型文件下载 https://huggingface.co/models

'''

# bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
# tokenizer = BertTokenizer.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained(r"/Users/mac/Documents/bert-base-chinese")

string = "咱呀么老百姓今儿个真高兴"
#分字
tokens = tokenizer.tokenize(string)
print("分字：", tokens)
#编码，前后自动添加了[cls]和[sep]
encoding = tokenizer.encode(string)
print("编码：", encoding)
# 编码： [101, 1493, 1435, 720, 5439, 4636, 1998, 791, 1036, 702, 4696, 7770, 1069, 102]
#文本对编码, 形式[cls] string1 [sep] string2 [sep]
string1 = "今天天气真不错"
string2 = "明天天气怎么样"
encoding = tokenizer.encode(string1, string2)
print("文本对编码：", encoding)
#同时输出attention_mask和token_type编码
encoding = tokenizer.encode_plus(string1, string2)
print("全部编码：", encoding)
encoding = tokenizer.encode(string, max_length=10, pad_to_max_length=True)
print("长度不够只有10个字符看看输出什么：", encoding)
encoding = tokenizer.encode("咱呀么", max_length=10, pad_to_max_length=True)
# [101, 1493, 1435, 720, 5439, 4636, 1998, 791, 1036, 102]
print("长度很短看看输出什么：", encoding)
# [101, 1493, 1435, 720, 102, 0, 0, 0, 0, 0]
encoding = tokenizer.encode(string, max_length=10, padding="max_length", truncation=True)
print("string看看输出什么：", encoding)
encoding = tokenizer.encode("咱呀么", max_length=10, padding="max_length", truncation=True)
print("truncation看看输出什么：", encoding)
# [101, 1493, 1435, 720, 102, 0, 0, 0, 0, 0]





