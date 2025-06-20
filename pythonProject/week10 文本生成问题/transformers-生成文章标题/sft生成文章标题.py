import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

"""
基于Bert结构，进行sft形式的训练
"""

class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1) # 输出的label为-1时，不计算交叉熵

    def forward(self, x, mask=None, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)
        
#加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus(path):
    corpus = []
    with open(path, encoding='utf8') as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["content"], line['title']])
    return corpus

def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        #构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))
        #padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def create_mask(s1, s2):
    len_s1 = s1 + 2
    len_s2 = s2 + 1 
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # s1 不会看到 s2的内容
    for i in range(len_s1):
        mask[i, len_s1:] = 0
    for i in range(len_s2):
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    
    return mask

def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end-w_start]
    return result

def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)

    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
    
def main(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    max_length = 50     #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率

    pretrain_model_path = r'/Users/mac/Documents/bert-base-chinese'

    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, mask, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        # print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))

        query_content = '中国新闻周刊独家私信报道：天价却当白菜抢的时代一去不复返。声声从紧“杀奢令”，层出不穷“奢负面”，让奢侈品从“中国梦”中惊醒。'
        print(generate_sentence(query_content, model, tokenizer))
    
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", 'pth')
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

                        
if __name__ == "__main__":
    main("sample_data.json", False)

                        
    
        

