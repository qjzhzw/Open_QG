### Dataset类

import torch
import torch.utils.data
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, mode='train'):
        '''
        data: 加载了的pt文件的内容
        mode: train表示是训练集的Dataset类构造
              dev表示是验证集的Dataset类构造
        '''

        # 从data中读取所有信息
        self.mode = mode
        self.vocab = data['vocab']
        self.train_input_indices = data['train_input_indices']
        self.train_output_indices = data['train_output_indices']
        self.dev_input_indices = data['dev_input_indices']
        self.dev_output_indices = data['dev_output_indices']

        # 断言: 训练集/验证集的输入输出数量必须一致
        assert len(self.train_input_indices) == len(self.train_output_indices)
        assert len(self.dev_input_indices) == len(self.dev_output_indices)

    # 这个方法必须实现,返回每个batch的内容
    def __getitem__(self, index):
        # 断言: mode值一定在['train', 'dev']范围内
        assert self.mode in ['train', 'dev']

        # 根据mode值的不同返回不同的内容
        if self.mode == 'train':
            return self.train_input_indices[index], self.train_output_indices[index], self.vocab
        elif self.mode == 'dev':
            return self.dev_input_indices[index], self.dev_output_indices[index], self.vocab

    # 这个方法必须实现,否则会报错:NotImplementedError
    def __len__(self):
        return len(self.train_input_indices)


def collate_fn(data):
    '''
    data: 输入的数据(即上面的__getitem__传下来的),原始的一个batch的内容
          是一个二维list,第一维是batch_size,第二维是batch中每个句子(不等长)
    return batch_input: 输入序列的batch
    return batch_output: 输出序列的batch
    '''

    # 对模型的输入序列和输出序列分别构造batch
    batch_input = get_batch(data, mode=0)
    batch_output = get_batch(data, mode=1)

    return batch_input, batch_output


def get_batch(data, mode=0):
    '''
    data: 输入的数据(即上面的__getitem__传下来的),原始的一个batch的内容
          是一个二维list,第一维是batch_size,第二维是batch中每个句子(不等长)
    mode: 0表示是输入序列
          1表示是输出序列
    return batch: 构造好的batch
    '''

    # 获得该batch中的最大句长
    max_len = 0
    # 这里的single_data是上面的__getitem__传下来的,原始的一个batch的内容
    # 获得single_data以后,要通过索引的方式获取其中每个需要的部分
    # 例如single_data[0]是输入序列,single_data[1]是输出序列
    for single_data in data:
        sentence = single_data[mode]
        if len(sentence) > max_len:
            max_len = len(sentence)

    # 构造batch,不足的部分用<pad>对应的索引填充
    batch = []
    for single_data in data:
        sentence = single_data[mode]
        vocab = single_data[-1]
        batch_sentence = sentence + [vocab.word2index['<pad>']] * (max_len - len(sentence))
        batch.append(batch_sentence)

    # 将二维list的batch直接转换为tensor的形式
    batch = torch.tensor(batch)

    return batch
