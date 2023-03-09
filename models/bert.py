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
        self.linear_size = 256



class Model(nn.Module):

    def __init__(self, config):

        def create_encoder():
            encoder_layer1 = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            encoder_layer2 = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            encoder = nn.Sequential(encoder_layer1, encoder_layer2)
            return encoder

        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.main_encoder = nn.Sequential(create_encoder(), create_encoder())
        self.claim_encoder = create_encoder()
        self.complaint_encoder = create_encoder()
        self.answer_encoder = create_encoder()

        self.linear1 = nn.Linear(config.hidden_size*3, config.linear_size)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(config.linear_size, config.num_classes)



    def forward(self, claim, complaint, answer):
        def main_forward(x, specific_model):
            context = x[:, 0, :]  # 输入的句子
            mask = x[:, 1, :]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
            bert_output, _ = self.bert(context, attention_mask=mask)
            bert_output = bert_output[-1]
            encoder_outputs = self.main_encoder(bert_output)
            specific_outputs = specific_model(encoder_outputs)
            pooled_output = torch.mean(specific_outputs, dim=1)
            return pooled_output

        claim = main_forward(claim, self.claim_encoder)
        complaint = main_forward(complaint, self.complaint_encoder)
        answer = main_forward(answer, self.answer_encoder)

        all = torch.cat((claim, complaint, answer), 1)

        out = self.linear1(all)
        out = self.sigmoid(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
