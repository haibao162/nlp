import jieba
import math
import os
import random
import re
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk


"""
基于tfidf实现简单文本摘要
"""

jieba.initialize()


#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
def load_data(file_path):
    corpus = []
    with open(file_path, encoding='utf8') as f:
        documents = json.loads(f.read())
        for document in documents:
            assert "\n" not in document['title']
            assert "\n" not in document['content']
            corpus.append(document['title'] + '\n' + document['content'])
        tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict, corpus

#计算每一篇文章的摘要
#输入该文章的tf_idf词典，和文章内容
#top为人为定义的选取的句子数量
#过滤掉一些正文太短的文章，因为正文太短在做摘要意义不大
def generate_document_abastract(document_tf_idf, document, top=3):
    sentences = re.split("？|！|。", document)
    #过滤掉正文在五句以内的文章
    if len(sentences) <= 5: 
        return None
    result = []
    for index, sentence in enumerate(sentences):
        sentence_score = 0
        words = jieba.lcut(sentence)
        # 计算句子包含的关键词的总得分，得分越高，说明该句子越应该作为关键句
        for word in words:
            sentence_score += document_tf_idf.get(word, 0)
        sentence_score /= (len(words) + 1)
        result.append([sentence_score, index])
    result = sorted(result, key=lambda x:x[0], reverse=True)
    important_sentence_indexs = sorted([x[1] for x in result[:top]])
    return "。".join([sentences[index] for index in important_sentence_indexs])
        

def generate_abstract(tf_idf_dict, corpus):
    res = []
    # print(tf_idf_dict, 'tf_idf_dict')
    # '不想': 0.004931657024216492, '朋友': 0.003322067713620321,
    for index, document_tf_idf in tf_idf_dict.items():
        title, content = corpus[index].split('\n')
        abstract = generate_document_abastract(document_tf_idf, content)
        if abstract is None:
            continue
        corpus[index] += "\n" + abstract
        res.append({"标题": title, "正文": content, '摘要': abstract})
    
    return res

if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    res = generate_abstract(tf_idf_dict, corpus)
    writer = open("abstract.json", "w", encoding="utf8")
    writer.write(json.dumps(res, ensure_ascii=False, indent=2))
    writer.close()


