import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from collections import defaultdict

'''
词向量模型的简单实现
'''
#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#corpus: [["今天", "天气", "不错"], ["你", "好", "吗"]]
#dim指定词向量的维度，如100

# sg为3是Skip-Gram模型
def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    print(model, 'model')
    # Word2Vec<vocab=19322, vector_size=128, alpha=0.025> model
    model.save('model3.w2v')
    return model

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def main():
    sentences = []
    with open('corpus.txt', encoding='utf8') as f:
        for line in f:
            sentences.append(jieba.lcut(line))
    model = train_word2vec_model(sentences, 128)
    return model

if __name__ == "__main__":
    model = main()
    print(model.wv, 'model.wv') 
    # KeyedVectors<vector_size=128, 19322 keys> model.wv
    # model = load_word2vec_model("model3.w2v") #加载
    res = model.wv.most_similar(positive=["男人", "母亲"], negative=["女人"])
    print(res)
    print(model.wv, 'model.wv222222') # KeyedVectors<vector_size=128, 19322 keys> model.wv222222





# 获取结巴分词Skip-Gram 模型 来处理这样的语料库，自然而然会出现准确度太低的原因
# 我们只需要在文本预处理时不再对文本进行去停用词、根据词性筛选特定词，就可保留文本语义。
def get_split(file_path, car_path, corpus_path):
    split = ''
 
    # 标点符号
    remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
 
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()
        # 去除标点符号
        txt = re.sub(remove_chars, "", txt)
 
        # 增加专业名词
        jieba.load_userdict(car_path)
        words = [w for w in jieba.cut(txt, cut_all=False)]
        text = ' '.join(words)
        split += text
 
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(split)

                        