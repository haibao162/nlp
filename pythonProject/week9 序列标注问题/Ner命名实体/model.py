
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # bidirectional: 双向LSTM
        # LSTM和RNN的输出完全一致，将第三维的hidden_size变成LSTM的hidden_size*2
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True) # 第一维是batch_size
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss采用交叉熵损失

    def forward(self, x, target=None):
        x = self.embedding(x) # batch_size, sen_len -> batch_size, sen_len, hidden_size
        x, _ = self.layer(x) # batch_size, sen_len, hidden_size * 2
        predict = self.classify(x)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict
            
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    
if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 100
    model = TorchModel(Config)
    x = model(torch.LongTensor([[1,2,3], [4,5,6]]))
    print(111)
    print(x.shape) # torch.Size([2, 3, 9])