### 训练器

import argparse
import logging
import json
import os
import torch
import torch.utils.data

from vocab import Vocab
from dataset import Dataset, collate_fn

logger = logging.getLogger()


def prepare_dataloaders(data, params):
    '''
    data: 输入的数据
    params: 参数集合
    return train_loader: 训练集的dataloader
    return dev_loader: 训练集的dataloader
    '''

    train_loader = torch.utils.data.DataLoader(
        Dataset(data),
        num_workers = 0,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = True
    )

    return train_loader


if __name__ == '__main__':

    # logger的一些设置
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s:  %(message)s ', '%Y/%m/%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_data_dir', type=str, default='data', help='数据主目录')
    parser.add_argument('--data_dir', type=str, default='squad', help='需要处理的数据所在的子目录')
    parser.add_argument('--input_dir', type=str, default='output.pt', help='输入的pt文件位置')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size')
    params = parser.parse_args()

    input_dir = os.path.join(params.main_data_dir, params.data_dir, params.input_dir)

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(input_dir)

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader = prepare_dataloaders(data, params)

    for i, data in enumerate(train_loader):
        if i==0:
            print(data)
            break
