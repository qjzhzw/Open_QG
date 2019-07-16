### Dataset类

import torch
import torch.utils.data
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.vocab = data['vocab']
        self.train_input_indices = data['train_input_indices']
        self.train_output_indices = data['train_output_indices']
        assert len(self.train_input_indices) == len(self.train_output_indices)

    def __getitem__(self, index):
        return self.train_input_indices[index], self.vocab

    # 这个方法必须实现,否则会报错:NotImplementedError
    def __len__(self):
        return len(self.train_input_indices)

# 构造每个batch
def collate_fn(data):
    '''
    data: 输入的数据(即上面的__getitem__传下来的),原始的一个batch的内容
    '''

    # data是一个二维list,第一维是batch_size,第二维是batch中每个句子(不等长)
    batch_size = len(data)

    # 获得该batch中的最大句长
    max_len = 0
    # 这里的single_data是上面的__getitem__传下来的,原始的一个batch的内容
    # 获得single_data以后,要通过索引的方式获取其中每个需要的部分
    for single_data in data:
        sentence = single_data[0]
        if len(sentence) > max_len:
            max_len = len(sentence)

    # 构造batch,不足的部分用<pad>对应的索引填充
    batch = []
    for single_data in data:
        sentence = single_data[0]
        vocab = single_data[-1]
        batch_sentence = sentence + [vocab.word2index['<pad>']] * (max_len - len(sentence))
        batch.append(batch_sentence)

    # 将二维list的batch直接转换为tensor的形式
    batch = torch.tensor(batch)

    return batch
