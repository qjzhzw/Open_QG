### 训练器

import argparse
import logging
import json
import os
import torch
import torch.utils.data

from vocab import Vocab
from dataset import Dataset, collate_fn
from model import Model

logger = logging.getLogger()


def prepare_dataloaders(data, params):
    '''
    data: 输入的数据
    params: 参数集合
    return train_loader: 训练集的dataloader
    return dev_loader: 训练集的dataloader
    '''

    train_loader = torch.utils.data.DataLoader(
        Dataset(data, mode='train'),
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = True
    )

    logger.info('正在构造train_loader')

    dev_loader = torch.utils.data.DataLoader(
        Dataset(data, mode='dev'),
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = False
    )

    logger.info('正在构造dev_loader')

    return train_loader, dev_loader


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
    parser.add_argument('--num_workers', type=int, default=0, help='模型超参数:num_workers(DataLoader中设置)')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size(DataLoader中设置)')
    params = parser.parse_args()

    input_dir = os.path.join(params.main_data_dir, params.data_dir, params.input_dir)

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(input_dir)

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader, dev_loader = prepare_dataloaders(data, params)

    # 开始写训练器部分
    for index, data in enumerate(train_loader):
        model = Model()
        output = model(data)
        # optimizer.zero_grad()
        # loss = f(data, output)
        # loss.backward()
