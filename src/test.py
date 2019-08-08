#!/usr/bin/env python
# encoding: utf-8
'''
数据测试:
(1)将模型测试集的输入输出数据构造为batch
(2)将测试集数据输入模型,进行模型测试
(3)保存模型测试的预测结果
'''
__author__ = 'qjzhzw'

import json
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from logger import logger
from params import params
from vocab import Vocab
from dataset import Dataset, collate_fn
from model import Model
from beam import Generator


def prepare_dataloaders(params, data):
    '''
    作用:
    将模型测试集的输入输出数据构造为batch

    输入参数:
    params: 参数集合
    data: 输入的数据

    输出参数:
    test_loader: 测试集的dataloader
    '''

    logger.info('正在从{}中读取数据'.format(params.dataset_dir))

    # 构造test_loader
    test_dataset = Dataset(data, mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = False
    )
    logger.info('正在构造test_loader,共有{}个batch'.format(len(test_loader)))

    return test_loader


def test_model(params, vocab, test_loader):
    '''
    作用:
    测试模型

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    test_loader: 测试集的dataloader
    '''

    logger.info('正在加载模型,即将开始测试')

    # 定义模型
    model = Model(params, vocab).to(params.device)

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 加载模型参数
    if os.path.exists(params.checkpoint_file):
        model_params = torch.load(params.checkpoint_file, map_location=params.device)
        model.load_state_dict(model_params)
        logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))
    else:
        logger.info('注意!!!没有训练好的模型参数,正在使用随机初始化模型参数进行测试')

    # 一轮模型测试
    sentences_pred = one_epoch(params, vocab, test_loader, model)

    # 将预测结果存入本地文件
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    # 依次写入文件
    f_pred = open(params.pred_file, 'w')
    for sentence_pred in sentences_pred:
        f_pred.write(sentence_pred + '\n')
    f_pred.close()
    logger.info('测试阶段的预测结果已经保存至{}'.format(params.pred_file))

    # 原始的真实输出文件,需要从数据目录移动到输出目录下,用于和预测结果进行比较
    shutil.copyfile(params.test_question_file, params.gold_file)

    # 使用multi-bleu.perl的脚本对结果进行评估
    os.system('evaluate/multi-bleu.perl %s < %s' %(params.gold_file, params.pred_file))


def one_epoch(params, vocab, loader, model):xwxw
    '''
    作用:
    每一轮的测试

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    loader: 测试集的dataloader
    model: 当前使用模型

    输出参数:
    sentences_pred: 测试结束时得到的预测序列集合
    '''

    logger.info('测试阶段')
    model.eval()

    # 我们保存所有得到的输出序列
    sentences_pred = []

    # 每一个batch的测试
    for batch_index, batch in enumerate(tqdm(loader)):
        # 从数据中读取模型的输入和输出
        input_indices = batch[0].to(params.device)
        output_indices = batch[1].to(params.device)
        # input_indices: [batch_size, input_seq_len]
        # output_indices: [batch_size, output_seq_len]

        # 使用beam_search算法,以<s>作为开始符得到完整的预测序列
        generator = Generator(params, model)
        indices_pred, scores_pred = generator.generate_batch(input_indices)
        # indices_pred: [batch_size, beam_size, output_seq_len]

        # 输出预测序列
        for indices in indices_pred:
            # indices[0]表示beam_size中分数最高的那个输出
            sentence = vocab.convert_index2sentence(indices[0])
            sentences_pred.append(' '.join(sentence))

        # 为了便于测试,在测试阶段也可以把预测序列打印出来
        if params.print_results:
            input_gold = ' '.join(vocab.convert_index2sentence(input_indices[-1]))
            output_gold = ' '.join(vocab.convert_index2sentence(output_indices[-1]))
            output_pred = sentences_pred[-1]
            logger.info('真实输入序列 : {}'.format())
            logger.info('预测输出序列 : {}'.format(sentence))

    return sentences_pred


if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']
    params = data['params']

    params.load_model = True
    params.max_seq_len = 30

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    test_loader = prepare_dataloaders(params, data)

    # 测试模型
    test_model(params, vocab, test_loader)
