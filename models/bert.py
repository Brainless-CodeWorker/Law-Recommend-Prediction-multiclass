# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.jsonl'                                # 训练集
        self.dev_path = dataset + '/data/val.jsonl'                                    # 验证集
        self.test_path = dataset + '/data/test.jsonl'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', errors='ignore').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 1                                             # mini-batch大小
        self.pad_size = 512                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(weights * x, dim=1)
        return context_vector

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.self_attention = SelfAttention(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens):
        def main_forward(x):
            context = x[:, 0, :]  # 输入的句子
            mask = x[:, 1, :]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
            _, pooled = self.bert(context, attention_mask=mask)
            return pooled

        outputs = []
        for token in tokens:
            out = main_forward(token)
            outputs.append(out)
        out_tensors = torch.stack(outputs, dim=0)

        attention_outputs = self.self_attention(out_tensors)
        out = self.linear(attention_outputs)
        out = self.sigmoid(out)
        out = out[:, 0]
        return out
