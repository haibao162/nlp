# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data, load_vocab
from model import TorchModel, choose_optimizer
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pos = 0
class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.pos = pos

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # print(len(self.valid_data), 'self.valid_data')
            # if self.pos == 0:
            #     print(batch_data[0][2], 'batch_data')
                # 中 共 中 央 政 治 局 委 员 、 书 记 处 书 记 丁 关 根 主 持 今 天 的 座 谈 会 。
            self.pos = self.pos + 1
            
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            # if self.pos == 1:
            #     print(sentences[2], 'sentence')
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id) #不输入labels，使用模型当前参数进行预测
                # if self.pos == 1:
                #     print(pred_results[2], input_id[2], 'pred_results[2]')
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return
    
    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
            # 如果使用交叉熵的话，预测的输入是batch_size * max_length * class_num,class_num里最大的权重就是预测值。
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist() # 转成list数组
            true_label = true_label.cpu().detach().tolist()
            # print(true_label, 'true_label')
            

            true_entities = self.decode(sentence, true_label) # 真实值
            # if self.config["model_type"] == "bert":
            #     pred_label = pred_label[1:-1] # 如果用bert预测值要去掉token
            #     true_label = true_label[1:-1] # 会丢掉前后两个
            #     print(len(pred_label), 'pred_label')
            #     print(len(true_label), 'true_label')
            pred_entities = self.decode(sentence, pred_label)
            # if self.pos == 1:
            #     print(true_entities, pred_entities,'pred_entities')
                
                # defaultdict(<class 'list'>, {}) defaultdict(<class 'list'>, {}) pred_entities
                # defaultdict(<class 'list'>, {'LOCATION': ['中国'], 'PERSON': ['邓小平']}) defaultdict(<class 'list'>, {'PERSON': ['邓小平', '毛泽东']}) pred_entities
                # defaultdict(<class 'list'>, {'ORGANIZATION': ['中共中央政治局'], 'PERSON': ['丁关根'], 'TIME': ['今天']}) defaultdict(<class 'list'>, {'ORGANIZATION': ['中共中央政治局'], 'TIME': ['今天']}) pred_entities
                # defaultdict(<class 'list'>, {'PERSON': ['丁关根', '胡锦涛', '邓小平', '邓小平']}) defaultdict(<class 'list'>, {'PERSON': ['丁关', '胡锦涛']}) pred_entities


            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        # 使用bert需要加"$"
        sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span() # 获取实体命名的位置
            results["LOCATION"].append(sentence[s:e]) # 获取结果，预测值和真实值比较
        for location in re.finditer("(15+)", labels):
            s, e = location.span() # 获取实体命名的位置
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span() # 获取实体命名的位置
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span() # 获取实体命名的位置
            results["TIME"].append(sentence[s:e])
        return results
                        
if __name__ == "__main__":
    from config import Config
    vocab = load_vocab(Config["vocab_path"]) # chars.txt
    Config["vocab_size"] = len(vocab)
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model_output/epoch_20.pth"))
    evaluator = Evaluator(Config, model, logger)
    evaluator.eval(1)

    # dg = DataGenerator("ner_data/train", Config)
    # dg2 = load_data("ner_data/test", Config)
    # dg2 = load_data(Config["valid_data_path"], Config, shuffle=False)
    # print(dg2[0])
        
# 不加dropout
# 2025-05-09 03:02:04,499 - __main__ - INFO - PERSON类实体，准确率：0.615385, 召回率: 0.373057, F1: 0.464511
# 2025-05-09 03:02:04,499 - __main__ - INFO - LOCATION类实体，准确率：0.745856, 召回率: 0.564854, F1: 0.642852
# 2025-05-09 03:02:04,499 - __main__ - INFO - TIME类实体，准确率：0.837662, 召回率: 0.724719, F1: 0.777103
# 2025-05-09 03:02:04,499 - __main__ - INFO - ORGANIZATION类实体，准确率：0.470588, 召回率: 0.252632, F1: 0.328763

# 自己加了线性层
# 2025-05-09 11:00:30,492 - __main__ - INFO - PERSON类实体，准确率：0.707317, 召回率: 0.448454, F1: 0.548891
# 2025-05-09 11:00:30,493 - __main__ - INFO - LOCATION类实体，准确率：0.673171, 召回率: 0.577406, F1: 0.621617
# 2025-05-09 11:00:30,493 - __main__ - INFO - TIME类实体，准确率：0.891892, 召回率: 0.741573, F1: 0.809811
# 2025-05-09 11:00:30,493 - __main__ - INFO - ORGANIZATION类实体，准确率：0.612500, 召回率: 0.515789, F1: 0.559995