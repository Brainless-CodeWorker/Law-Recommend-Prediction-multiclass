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
        self.batch_size = 2                                             # mini-batch大小
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
        self.seq_len = config.pad_size
        self.self_attention = SelfAttention(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def main_forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(dim=0)
        sentence_sum = x.size(dim=1)
        context = x[:, :, 0, :]  # 输入的句子
        mask = x[:, :, 1, :]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        context = context.view(-1, self.seq_len)
        mask = mask.view(-1, self.seq_len)
        _, pooled = self.bert(context, attention_mask=mask)
        pooled = pooled.view(batch_size, sentence_sum, pooled.size(dim=1))
        return pooled

    def main_forward_end(self, out_tensors):
        attention_outputs = self.self_attention(out_tensors)
        out = self.linear(attention_outputs)
        out = self.sigmoid(out)
        return out.view(2)

    #预先求一次所有的tensor
    def pre_forward(self, tokens):
        self.output_list = self.main_forward(tokens)
        out = self.main_forward_end(self.output_list)
        return out

    #真·正向传播
    def forward(self, token, id):
        new_out = self.output_list.clone()
        out = self.main_forward(token)
        new_out = new_out.permute(1, 0, 2)
        new_out[id] = out[0]
        new_out = new_out.permute(1, 0, 2)
        out = self.main_forward_end(new_out)
        return out
