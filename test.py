### 测试器

import argparse
import logging
import json
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from vocab import Vocab
from dataset import Dataset, collate_fn
from model import Model

logger = logging.getLogger()


def prepare_dataloaders(data, params):
    '''
    data: 输入的数据
    params: 参数集合
    return test_loader: 测试集的dataloader
    '''

    # 构造test_loader
    test_dataset = Dataset(data, mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = True
    )
    logger.info('正在构造test_loader,共有{}个batch'.format(len(test_loader)))

    return test_loader


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
    parser.add_argument('--dataset_dir', type=str, default='squad', help='任务所使用的数据集所在的子目录')
    parser.add_argument('--temp_pt_file', type=str, default='data.pt', help='输入的pt文件位置')
    parser.add_argument('--num_workers', type=int, default=0, help='模型超参数:num_workers(DataLoader中设置)')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size(批训练大小,DataLoader中设置)')
    params = parser.parse_args()

    params.temp_pt_file = os.path.join(params.main_data_dir, params.dataset_dir, params.temp_pt_file)

    # 打印参数列表
    logger.info('参数列表:{}'.format(params))

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    test_loader = prepare_dataloaders(data, params)
