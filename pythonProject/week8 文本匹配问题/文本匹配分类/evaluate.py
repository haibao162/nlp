
# -*- coding: utf-8 -*-
from loader import load_data, load_vocab
from config import Config
import torch

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0} #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0} #清空上一轮的测试结果
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # print(batch_data, 'batch_data')
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        self.show_stats()
        return
    
    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label) # 获取预测值的位置
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return

if __name__ == "__main__":

    # 以下为测试代码
    import logging
    from model import TorchModel, choose_optimizer
    
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    vocab = load_vocab(Config["vocab_path"])
    Config["vocab_size"] = len(vocab)
    Config["class_num"] = 11

    #加载模型
    model = TorchModel(Config)
    evaluator = Evaluator(Config, model, logger)
    evaluator.eval(1)
