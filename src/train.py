#!/usr/bin/env python
# encoding: utf-8
'''
数据训练:
(1)将模型训练集/验证集的输入输出数据构造为batch
(2)将训练集数据输入模型,进行模型训练
(3)将验证集数据输入模型,进行模型验证
(4)根据验证集损失最小的原则,选择训练最好的模型参数进行保存
'''
__author__ = 'qjzhzw'

import json
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from logger import logger
from params import params
from vocab import Vocab
from dataset import Dataset, collate_fn
from optimizer import Optimizer


def prepare_dataloaders(params, data):
    '''
    作用:
    将模型训练集/验证集的输入输出数据构造为batch

    输入参数:
    params: 参数集合
    data: 输入的数据

    输出参数:
    train_loader: 训练集的dataloader
    dev_loader: 验证集的dataloader
    '''

    logger.info('正在从{}中读取数据'.format(params.dataset_dir))

    # 构造train_loader
    train_dataset = Dataset(params, data, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = True
    )
    logger.info('正在构造train_loader,共有{}个batch'.format(len(train_dataset)))

    # 构造dev_loader
    dev_dataset = Dataset(params, data, mode='dev')
    dev_loader = torch.utils.data.DataLoader(
        dataset = dev_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = False
    )
    logger.info('正在构造dev_loader,共有{}个batch'.format(len(dev_dataset)))

    return train_loader, dev_loader


def train_model(params, vocab, train_loader, dev_loader):
    '''
    作用:
    训练和验证模型

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    train_loader: 训练集的dataloader
    dev_loader: 验证集的dataloader
    '''

    logger.info('正在加载模型,即将开始训练')

    # 定义模型
    model = Model(params, vocab).to(params.device)

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 如果参数中设置了加载训练好的模型参数且模型参数文件存在,则加载模型参数
    if params.load_model and os.path.exists(params.checkpoint_file):
        model_params = torch.load(params.checkpoint_file, map_location=params.device)
        model.load_state_dict(model_params)
        logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))
    else:
        logger.info('没有训练好的模型参数,从头开始训练')

    # 定义优化器
    optimizer = Optimizer(params, model)

    # 存储每一轮验证集的损失,根据验证集损失最小来挑选最好的模型进行保存
    total_loss_epochs = []

    # 每一轮的训练和验证
    for epoch in range(1, params.num_epochs + 1):
        # 一轮模型训练
        model, _, _ = one_epoch(params, vocab, train_loader, model, optimizer, epoch, mode='train')
        # 一轮模型验证
        model, sentences_pred, total_loss = one_epoch(params, vocab, dev_loader, model, optimizer, epoch, mode='dev')

        # 存储每一轮验证集的损失
        total_loss_epochs.append(total_loss)
        logger.info('第{}轮的验证集的损失为{},当前最好模型的损失为{}'.format(epoch, total_loss, min(total_loss_epochs)))

        # 将训练好的模型参数存入本地文件
        if not os.path.exists(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        # 根据验证集损失最小来挑选最好的模型进行保存
        if total_loss == min(total_loss_epochs):
            torch.save(model.state_dict(), params.checkpoint_file)
            logger.info('第{}轮的模型参数已经保存至{}'.format(epoch, params.checkpoint_file))


def one_epoch(params, vocab, loader, model, optimizer, epoch, mode='train'):
    '''
    作用:
    每一轮的训练/验证

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    loader: 训练集/验证集的dataloader
    model: 当前使用模型
    optimizer: 当前使用优化器
    epoch: 当前轮数
    mode: train表示是训练阶段
          dev表示是验证阶段

    输出参数:
    model: 训练/验证结束时得到的模型
    sentences_pred: 验证结束时得到的预测序列集合(只有验证时有内容,训练时为空list)
    total_loss: 总损失
    '''

    # 断言: mode值一定在['train', 'dev']范围内
    assert mode in ['train', 'dev']
    if mode == 'train':
        logger.info('训练阶段,第{}轮'.format(epoch))
        model.train()
    elif mode == 'dev':
        logger.info('验证阶段,第{}轮'.format(epoch))
        model.eval()

    # 对于验证阶段,我们保存所有得到的输出序列
    sentences_pred = []
    # 记录训练/验证的总样例数
    total_examples = 0
    # 记录训练/验证的总损失
    total_loss = 0

    # 每一个batch的训练/验证
    for batch_index, batch in enumerate(tqdm(loader)):
        # 从数据中读取模型的输入和输出
        input_indices = batch[0].to(params.device)
        output_indices = batch[1].to(params.device)
        if torch.is_tensor(batch[2]):
            answer_indices = batch[2].to(params.device)
        else:
            answer_indices = None
        # input_indices: [batch_size, input_seq_len]
        # output_indices: [batch_size, output_seq_len]
        # answer_indices: [batch_size, output_seq_len]

        # 模型:通过模型输入来预测真实输出,即
        #  <s>  1   2   3
        #  --------------->
        #   1   2   3  </s>
        # 真实输出是在原始数据的基础上"去头"(decoder部分的输出)
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: 1 2 3 </s>
        output_indices_gold = output_indices[:, 1:]
        # 模型输入是在原始数据的基础上"去尾"(decoder部分的输入)
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: <s> 1 2 3
        output_indices = output_indices[:, :-1]

        # 将输入数据导入模型,得到预测的输出数据
        output_indices_pred = model(input_indices, output_indices, answer_indices=answer_indices)
        # output_indices_pred: [batch_size, output_seq_len, vocab_size]

        # 将基于vocab的概率分布,通过取最大值的方式得到预测的输出序列
        indices_pred = torch.max(output_indices_pred, dim=-1)[1]
        # indices_pred: [batch_size, output_seq_len]

        # 输出预测序列
        for indices in indices_pred:
            # full: True表示输出完整序列
            #       False表示遇到</s>就停止(只输出到</s>前的序列)
            sentence = vocab.convert_index2sentence(indices, full=False)
            sentences_pred.append(' '.join(sentence))

        # 利用预测输出和真实输出计算损失
        # softmax在模型中已经做了,因此还需要自己做一下log
        # output_indices_pred = F.log_softmax(output_indices_pred, dim=-1)
        output_indices_pred = torch.log(output_indices_pred)
        # output_indices_pred: [batch_size, output_seq_len, vocab_size]
        # output_indices_gold: [batch_size, output_seq_len]
        if params.label_smoothing:
            # 自己编写损失函数
            # 使用标签平滑归一化
            batch_size = output_indices_pred.size(0)
            output_seq_len = output_indices_pred.size(1)
            vocab_size = output_indices_pred.size(2)

            # 调整维度
            output_indices_pred = output_indices_pred.contiguous().view(batch_size * output_seq_len, vocab_size)
            output_indices_gold = output_indices_gold.contiguous().view(batch_size * output_seq_len).unsqueeze(1)
            # output_indices_pred: [batch_size * output_seq_len, vocab_size]
            # output_indices_gold: [batch_size * output_seq_len, 1]

            # 计算损失
            nll_loss = -output_indices_pred.gather(dim=-1, index=output_indices_gold)
            smooth_loss = -output_indices_pred.sum(dim=-1, keepdim=True)
            # nll_loss: [batch_size * output_seq_len]
            # smooth_loss: [batch_size * output_seq_len]

            # 通过取平均的方式得到损失
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()

            # 使用标签平滑归一化,得到最终损失
            eps_i = params.label_smoothing_eps / vocab_size
            loss = (1 - params.label_smoothing_eps) * nll_loss + eps_i * smooth_loss
        else:
            # 使用内置的损失函数
            # NLLLoss(x,y)的两个参数:
            # x: [batch_size, num_classes, ……], 类型为LongTensor, 是预测输出
            # y: [batch_size, ……], 类型为LongTensor, 是真实输出
            criterion = torch.nn.NLLLoss(ignore_index=vocab.word2index['<pad>'])

            output_indices_pred = output_indices_pred.permute(0, 2, 1)
            # output_indices_pred: [batch_size, vocab_size, output_seq_len]
            # output_indices_gold: [batch_size, output_seq_len]
            loss = criterion(output_indices_pred, output_indices_gold)

        # 计算到当前为止的总样例数和总损失
        num_examples = input_indices.size(0)
        total_examples += num_examples
        total_loss += loss.item() * num_examples

        # 如果参数中设置了打印模型损失,则打印模型损失
        if params.print_loss:
            logger.info('Epoch : {}, batch : {}/{}, loss : {}'.format(epoch, batch_index, len(loader), loss))

        # 如果是训练阶段,就利用优化器进行BP反向传播,更新参数
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 为了便于测试,在训练/验证阶段也可以把预测序列打印出来
        if params.print_results:
            input_gold = ' '.join(vocab.convert_index2sentence(input_indices[-1]))
            logger.info('真实输入序列 : {}'.format(input_gold))

            if torch.is_tensor(answer_indices):
                answer = answer_indices[-1] * input_indices[-1]
                answer = ' '.join(vocab.convert_index2sentence(answer, full=True))
                logger.info('真实答案序列 : {}'.format(answer))

            output_gold = ' '.join(vocab.convert_index2sentence(output_indices[-1]))
            logger.info('真实输出序列 : {}'.format(output_gold))

            output_pred = sentences_pred[-1]
            logger.info('预测输出序列 : {}'.format(output_pred))

    # 计算总损失
    total_loss = total_loss / total_examples

    return model, sentences_pred, total_loss


if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']
    params = data['params']

    # params.num_epochs = 5
    # params.print_loss = True
    # params.with_copy = True
    # params.share_embeddings = True

    if params.rnnsearch:
        from rnnsearch import Model
    else:
        from transformer import Model

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader, dev_loader = prepare_dataloaders(params, data)

    # 训练模型
    train_model(params, vocab, train_loader, dev_loader)
