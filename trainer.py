### 训练器

import argparse
import logging
import json
import os
import torch
import torch.nn as nn
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
    return dev_loader: 验证集的dataloader
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


def train_model(train_loader, dev_loader, vocab, params):
    '''
    train_loader: 训练集的dataloader
    dev_loader: 验证集的dataloader
    vocab: 从pt文件中读取的该任务所使用的vocab
    params: 参数集合
    '''

    logger.info('正在加载模型,即将开始训练')

    # 定义模型
    model = Model(params, vocab)
    # 使模型进入训练模式,即保留dropout等
    model.train()
    # 如果参数中设置了使用cuda且当前cuda可用,则将模型放到cuda上
    flag_cuda = False
    if params.cuda and torch.cuda.is_available():
        model.cuda()
        flag_cuda = True

    logger.info('当前模型cuda设置为{}'.format(flag_cuda))

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    # 定义损失函数
    # NLLLoss(x,y)的两个参数:
    # x: [batch_size, num_classes, ……], 类型为LongTensor, 是预测输出
    # y: [batch_size, ……], 类型为LongTensor, 是真实输出
    loss_func = torch.nn.NLLLoss()

    # 每一轮的训练
    for epoch in range(params.num_epochs):
        logger.info('训练第{}轮'.format(epoch))
        total_loss = 0
        # 每一个batch的训练
        for index, data in enumerate(train_loader):
            # 从数据中读取模型的输入和输出
            train_input_indices = data[0]
            train_output_indices = data[1]

            # 模型:通过模型输入来预测真实输出,即
            #  <s>  1   2   3
            #  --------------->
            #   1   2   3  </s>
            # 真实输出是在原始数据的基础上"去头"
            # 原始数据: <s> 1 2 3 </s>
            # 真实输出: 1 2 3 </s>
            train_output_indices_gold = train_output_indices[:, 1:]
            # 模型输入是在原始数据的基础上"去尾"(decoder部分的输入)
            # 原始数据: <s> 1 2 3 </s>
            # 真实输出: <s> 1 2 3
            train_output_indices = train_output_indices[:, :-1]

            # 将输入数据导入模型,得到预测的输出数据
            train_output_indices_pred = model(train_input_indices, train_output_indices)

            # 利用预测输出和真实输出计算损失
            loss = loss_func(train_output_indices_pred, train_output_indices_gold)
            total_loss += loss.item()
            logger.info('当前训练第{}个batch,损失为{}'.format(index, loss))

            # 利用优化器进行BP反向传播,更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用cuda')
    parser.add_argument('--num_workers', type=int, default=0, help='模型超参数:num_workers(DataLoader中设置)')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size(批训练大小,DataLoader中设置)')
    parser.add_argument('--learning_rate', type=int, default=0.1, help='模型超参数:learning_rate(学习率)')
    parser.add_argument('--num_epochs', type=int, default=10, help='模型超参数:num_epochs(训练轮数)')
    parser.add_argument('--embedding_size', type=int, default=512, help='模型超参数:embedding_size')
    params = parser.parse_args()

    input_dir = os.path.join(params.main_data_dir, params.data_dir, params.input_dir)

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(input_dir)
    vocab = data['vocab']

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader, dev_loader = prepare_dataloaders(data, params)

    # 训练模型
    train_model(train_loader, dev_loader, vocab, params)
