# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import json
import ast

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def token_and_pad(content, pad_size=512):
        token = config.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        return [token_ids, mask]

    def convert_label(s, n):
        a = s #ast.literal_eval(s)
        b = [0] * n
        for i in a:
            b[int(i)] = 1
        return b

    def load_dataset(path, pad_size=512):
        contents = []
        count = 100
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                count -= 1
                if count < 0:
                    break
                lin = line.strip()
                if not lin:
                    continue
                obj = json.loads(lin.strip())
                claim, complaint, answer, label = obj['claim'], obj['complaint'], obj['answer'], obj['laws']
                claim = token_and_pad(claim)
                complaint = token_and_pad(complaint)
                answer = token_and_pad(answer)
                label = convert_label(label, config.num_classes)
                contents.append((claim, complaint, answer, label))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev[:len(dev)//10], test[:len(dev)//10]


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        a = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        b = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        c = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        d = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return a, b, c, d.float()

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
