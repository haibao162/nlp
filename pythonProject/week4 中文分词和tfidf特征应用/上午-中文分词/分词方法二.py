#分词方法最大正向切分的第二种实现方式
import re
import time
import json

#加载词前缀词典
#用0和1来区分是前缀还是真词
#需要注意有的词的前缀也是真词，在记录时不要互相覆盖
def load_prefix_word_dict(path):
    prefix_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict:
                    prefix_dict[word[:i]] = 0
            prefix_dict[word] = 1
    return prefix_dict