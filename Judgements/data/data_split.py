'''
我现在有一个10000行的jsonl文件，请你将它切分为训练集，验证集和测试集，比例为9:0.5:0.5，请你写出对应的代码
'''

import json
import random

# 读取jsonl文件，将每行json字符串转换为对应的字典形式，并存储在列表中
def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 将列表按比例划分为训练集、验证集和测试集，并将它们写入对应的文件中
def split_data(data, train_ratio, val_ratio, test_ratio):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios should be equal to 1.0"
    random.shuffle(data)
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    return train_data, val_data, test_data

# 将数据写入jsonl文件
def write_jsonl_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 划分数据集
data = read_jsonl_file("劳动争议.jsonl")
train_data, val_data, test_data = split_data(data, 0.9, 0.05, 0.05)

# 将数据写入对应的文件
write_jsonl_file("train.jsonl", train_data)
write_jsonl_file("val.jsonl", val_data)
write_jsonl_file("test.jsonl", test_data)