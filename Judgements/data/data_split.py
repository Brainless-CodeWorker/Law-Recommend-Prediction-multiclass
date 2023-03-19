# -*- coding: utf-8 -*-
'''
我现在有一个10000行的jsonl文件，请你将它切分为训练集，验证集和测试集，比例为9:0.5:0.5，请你写出对应的代码
'''

import json
import random
import re

def split_sentences(text):
    # 使用正则表达式切分句子
    text = re.sub('[0-9]|\.+', '', text)
    sentences = re.split('[。；]', text)
    # 移除空字符串并过滤长度小于5的句子
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() and len(sentence.strip()) >= 5]
    for str in sentences:
        print(str)
    return sentences

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
    data = data[:len(data)//100]
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

#str='一、王秀丽与被告建立劳动关系的时间是2012年4月20日。2010年4月22日，王秀丽到被告处工作。在其入职申请中自己陈述从2009年到2010年4月期间，她本人在丰台区方庄做清洁及看楼等杂工。2009年4月20日，王秀丽根本就不在被告处工作。2012年2月10日，王秀丽本人由于身体原因，主动向被告提出辞职，经被告批准，她于当日离职，并办理了全部离职手续，解除了双方之间的劳动关系。2012年4月20日，王秀丽因找不到合适的工作，又到被告处申请工作，被告又一次善意的接受她，双方于当日签订了《劳动合同》，建立了劳动关系，王秀丽于当日办理了全部的入职手续，领取了工作服。随后，为了多取得一些实得收入，王秀丽于2012年5月23日书面主动要求被告不为其缴纳社会保险，但为了遵守劳动法，被告并没有同意其请求，依法为其缴纳了社会保险，并扣缴了她本人应当承担的金额。二、王秀丽双休日和法定假日加班不多，被告根本就没有拖欠王秀丽任何的工资和加班费。即便她在某周六或周日工作，但她在该周实际工作时间仅为36小时，并不符合加班的规定，被告没有义务支付其加班费。王秀丽即使在个别双休日上班，也通过轮休的方式加以补休。对于王秀丽无法补休双休日加班和法定假日的工作，被告已经按时足额的支付了其加班工资，根本就不存在拖欠。2015年8月8日，因王秀丽严重违反被告劳动纪律，本人对此也完全认可，被告通知其解除劳动关系。从8月11日，王秀丽就不再来上班。三、王秀丽因严重违反劳动合同和被告的规章制度被依法解除劳动关系，被告没义务向其支付解除劳动关系经济补偿金。2015年8月8日，王秀丽在收拾南楼411客房时，用涮完马桶的刷子直接涮洗脸盆和茶杯，被客人当场发现，客人根本无法接受该事实，当场投诉到被告，强烈要求饭店就王秀丽的行为给一个说法，并要求对此事做出处理，否则不接受任何条件。接到投诉后，经被告核实，客人投诉属实，王秀丽对此也没有异议，承认其不当行为。被告对客人进行多方的道歉和解释，被告免除了客人的全部房费和餐饮等费用，并向客人赠送了小礼品，客人才没有投诉到旅游和卫生监管部门，也没有向媒体披露，如果该事件被投诉或披露被媒体放大，后果对被告将不堪设想。如果客人投诉到旅游和卫生监管部门，被告将面临着停业整改，甚至是被摘除三星饭店的星级；如果客人通过媒体披露该事件，将直接影响到被告的声誉和客房入住率，严重损害被告的经济利益。双方签订的《劳动合同书》第十条第5款第3项“（王秀丽）被客人投诉，经查属实的，给甲方造成严重影响的”和第15项的“在工作过程中，违反甲方的管理规定，给甲方声誉造成负面影响的，丧失客户资源或遭到客户投诉的”约定，特别是第5款第15项约定，只要王秀丽遭到客户投诉的，被告就有权解除本合同，通知王秀丽后即生效，无须向其支付任何的经济补偿金或赔偿。事实上，王秀丽的行为不仅是遭到客户投诉，而且其行为给被告造成了严重影响。同时，《劳动合同书》第八条第4款也约定“乙方（指王秀丽）不得从事有损甲方形象、信誉及声誉的行为，不得谋取私利而损害甲方利益。乙方因故意或过失给甲方的形象、信誉及声誉造成重大影响的，甲方可根据饭店的规章制度，给予处分，直到解除劳动合同”。由上，被告有权予以辞退王秀丽，解除双方的劳动关系。《劳动合同书》第十四条约定，《员工守则》是本合同的附件，与本合同具有同等法律效力。根据《员工守则》第六章规定，员工行为构成D类的，被告有权解除劳动关系，予以辞退。根据D类第11项“做出任何破坏饭店声誉或损害饭店利益行为的”和14项“严重违反操作规程，造成严重后果的”的规定，王秀丽的行为已经破坏了被告的声誉，损害了被告的利益，违反了《员工守则》的规定，属于严重违纪过失，被告有权予以辞退王秀丽。2015年8月10日，被告作出了辞退王秀丽的决议，书面送达给王秀丽，王秀丽本人认可，并当场签字予以确认。同时被告依据饭店的规程，报总经理批准。2015年8月19日，被告再一次向王秀丽送达了书面通知，要求其到单位办理离职手续。四、王秀丽第4项仲裁请求是要求支付其解除劳动关系经济补偿金，而第5项请求却是要求支付其未提前解除劳动关系代通知金，逻辑关系混乱，实属无理。综上所述，恳请贵院依法驳回王秀丽的请求，以维护被告的合法权益。'
#split_sentences(str)

# 划分数据集
data = read_jsonl_file("劳动争议.jsonl")
train_data, val_data, test_data = split_data(data, 0.9, 0.05, 0.05)

# 将数据写入对应的文件
write_jsonl_file("train.jsonl", train_data)
write_jsonl_file("val.jsonl", val_data)
write_jsonl_file("test.jsonl", test_data)
