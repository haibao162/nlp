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

def cut_method2(string, prefix_dict):
    if string == "":
        return []
    words = [] #放入切好的词
    start_index, end_index = 0, 1
    window = string[start_index: end_index]
    find_word = window
    while start_index < len(string):
        # 没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            words.append(find_word)
            start_index += len(find_word)
            end_index = start_index + 1
            window = string[start_index:end_index]
            find_word = window
        #窗口是一个词
        elif prefix_dict[window] == 1:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = string[start_index:end_index]
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    return words

#cut_method是切割函数
#output_path是输出路径
def main(cut_method, input_path, output_path):
    word_dict = load_prefix_word_dict('dict.txt')
    # '高考状': 0, '高考状元': 1, '高考网': 1, '高考落': 0, '高考落榜': 1
    writer = open(output_path, 'w', encoding='utf8')
    start_time = time.time()
    with open(input_path, encoding='utf8') as f:
        for line in f:
            words = cut_method(line.strip(), word_dict)
            writer.write(" / ".join(words) + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return


string = "王羲之草书《平安帖》共有九行"
# string = "你到很多有钱人家里去看"
# string = "金鹏期货北京海鹰路营业部总经理陈旭指出"
# string = "伴随着优雅的西洋乐"
# string = "非常的幸运"
prefix_dict = load_prefix_word_dict("dict.txt")
# print(cut_method2(string, prefix_dict))
# print(json.dumps(prefix_dict, ensure_ascii=False, indent=2))
main(cut_method2, "corpus.txt", "cut_method2_output.txt")