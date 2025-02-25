import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema) # 几分类任务
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    questions = line["questions"]
                    label = line["target"]
                    label_index = torch.LongTensor([self.schema[label]]) #获取分类，是实际值
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        label_index = torch.LongTensor([self.schema[label]])
                        # input_id看做是onehot矩阵，表示输入的词向量，label_index表示输出的结果y
                        self.data.append([input_id, label_index])
                else:
                    # 验证集
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))

        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id
    
    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] -len(input_id))
        return input_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1 #0留给padding位置，所以从1开始
    return token_dict

#加载schema
# {
#   "停机保号": 0,
# }
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())
    
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    print(len(dg)) #1878 一份32，一共59份
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/valid.json", Config)
    # ../data/valid.json
    print(dg[1]) # q：办（540）理（2626）业务， target：宽泛业务问题
    load_data("../data/train.json", Config)