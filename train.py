### 训练器

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
    return train_loader: 训练集的dataloader
    return dev_loader: 验证集的dataloader
    '''

    # 构造train_loader
    train_dataset = Dataset(data, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = True
    )
    logger.info('正在构造train_loader,共有{}个batch'.format(len(train_dataset)))

    # 构造dev_loader
    dev_dataset = Dataset(data, mode='dev')
    dev_loader = torch.utils.data.DataLoader(
        dataset = dev_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = False
    )
    logger.info('正在构造dev_loader,共有{}个batch'.format(len(dev_dataset)))

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

    # 如果参数中设置了使用cuda且当前cuda可用,则将模型放到cuda上
    flag_cuda = False
    if params.cuda and torch.cuda.is_available():
        model.cuda()
        flag_cuda = True
    logger.info('当前模型cuda设置为{}'.format(flag_cuda))

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 如果参数中设置了加载训练好的模型参数且模型参数文件存在,则加载模型参数
    if params.load_model and os.path.exists(params.checkpoint_file):
        model_params = torch.load(params.checkpoint_file)
        model.load_state_dict(model_params)
        logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    # 每一轮的训练和验证
    for epoch in range(params.num_epochs):
        # 一轮模型训练
        model, _ = one_epoch(model, optimizer, epoch, train_loader, vocab, params, mode='train')
        # 一轮模型验证
        model, sentences_pred = one_epoch(model, optimizer, epoch, dev_loader, vocab, params, mode='dev')

        # 将预测结果存入本地文件
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir)
        f_pred = open(params.pred_file, 'w')
        for sentence_pred in sentences_pred:
            f_pred.write(sentence_pred + '\n')
        f_pred.close()
        logger.info('第{}轮的预测结果已经保存至{}'.format(epoch, params.pred_file))

        # 原始的真实输出文件,需要从数据目录移动到输出目录下,用于和预测结果进行比较
        shutil.copyfile(params.origin_gold_file, params.gold_file)

        # 使用multi-bleu.perl的脚本对结果进行评估
        os.system('evaluate/multi-bleu.perl %s < %s' %(params.gold_file, params.pred_file))

        # 将训练好的模型参数存入本地文件
        if not os.path.exists(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        torch.save(model.state_dict(), params.checkpoint_file)
        logger.info('第{}轮的模型参数已经保存至{}'.format(epoch, params.checkpoint_file))


def one_epoch(model, optimizer, epoch, loader, vocab, params, mode='train'):
    '''
    model: 当前使用模型
    optimizer: 当前使用优化器
    epoch: 当前轮数
    loader: 训练集/验证集的dataloader
    vocab: 从pt文件中读取的该任务所使用的vocab
    params: 参数集合
    mode: train表示是训练阶段
          dev表示是验证阶段
    return model: 训练/验证结束时得到的模型
    return sentences_pred: 验证结束时得到的预测序列集合(只有验证时有内容,训练时为空list)
    '''
    # 断言: mode值一定在['train', 'dev']范围内
    assert mode in ['train', 'dev']

    if mode == 'train':
        logger.info('训练第{}轮'.format(epoch))
        model.train()
    elif mode == 'dev':
        logger.info('验证第{}轮'.format(epoch))
        model.eval()

    # 对于验证阶段,我们保存所有得到的输出序列
    sentences_pred = []

    # 每一个batch的训练
    for batch_index, batch in enumerate(tqdm(loader)):
        # 从数据中读取模型的输入和输出
        input_indices = batch[0]
        output_indices = batch[1]

        # 如果参数中设置了使用cuda且当前cuda可用,则将数据放到cuda上
        if params.cuda and torch.cuda.is_available():
            input_indices = input_indices.cuda()
            output_indices = output_indices.cuda()

        # 模型:通过模型输入来预测真实输出,即
        #  <s>  1   2   3
        #  --------------->
        #   1   2   3  </s>
        # 真实输出是在原始数据的基础上"去头"
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: 1 2 3 </s>
        output_indices_gold = output_indices[:, 1:]
        # 模型输入是在原始数据的基础上"去尾"(decoder部分的输入)
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: <s> 1 2 3
        output_indices = output_indices[:, :-1]

        # 将输入数据导入模型,得到预测的输出数据
        output_indices_pred = model(input_indices, output_indices)

        # 定义损失函数
        # NLLLoss(x,y)的两个参数:
        # x: [batch_size, num_classes, ……], 类型为LongTensor, 是预测输出
        # y: [batch_size, ……], 类型为LongTensor, 是真实输出
        criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2index['<pad>'])

        # 利用预测输出和真实输出计算损失
        # output_indices_pred: [batch_size, vocab_size, output_seq_len]
        # output_indices_gold: [batch_size, output_seq_len]
        loss = criterion(output_indices_pred, output_indices_gold)

        # 如果参数中设置了打印模型损失,则打印模型损失
        if params.print_loss:
            logger.info('Epoch : {}, batch : {}/{}, loss : {}'.format(epoch, batch_index, len(loader), loss))

        # 如果是训练阶段,就利用优化器进行BP反向传播,更新参数
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 如果是验证阶段,给出预测的序列
        if mode == 'dev':
            # 将基于vocab的概率分布,通过取最大值的方式得到预测的输出序列
            output_indices_pred = output_indices_pred.permute(0, 2, 1)
            indices_pred = torch.max(output_indices_pred, dim=-1)[1]

            # 输出预测序列
            for indices in indices_pred:
                sentence = vocab.convert_index2sentence(indices)
                sentences_pred.append(' '.join(sentence))

    return model, sentences_pred


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
    parser.add_argument('--main_checkpoint_dir', type=str, default='checkpoint', help='输出的模型参数目录')
    parser.add_argument('--main_output_dir', type=str, default='output', help='输出的预测文件目录')
    parser.add_argument('--dataset_dir', type=str, default='squad', help='任务所使用的数据集所在的子目录')
    parser.add_argument('--temp_pt_file', type=str, default='data.pt', help='输入的pt文件位置')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pt', help='输出的模型参数位置')
    parser.add_argument('--pred_file', type=str, default='pred.txt', help='输出的预测文件位置')
    parser.add_argument('--gold_file', type=str, default='gold.txt', help='用于比较的真实文件位置')
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用cuda')
    parser.add_argument('--print_model', type=bool, default=True, help='是否打印出模型结构')
    parser.add_argument('--load_model', type=bool, default=False, help='是否加载训练好的模型参数')
    parser.add_argument('--print_loss', type=bool, default=True, help='是否打印出训练过程中的损失')
    parser.add_argument('--num_workers', type=int, default=0, help='模型超参数:num_workers(DataLoader中设置)')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size(批训练大小,DataLoader中设置)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='模型超参数:learning_rate(学习率)')
    parser.add_argument('--num_epochs', type=int, default=10, help='模型超参数:num_epochs(训练轮数)')
    parser.add_argument('--embedding_size', type=int, default=512, help='transformer模型超参数:embedding_size(词向量维度)')
    parser.add_argument('--num_layers', type=int, default=6, help='transformer模型超参数:num_layers')
    parser.add_argument('--num_heads', type=int, default=8, help='transformer模型超参数:num_heads')
    parser.add_argument('--d_model', type=int, default=512, help='transformer模型超参数:d_model')
    parser.add_argument('--d_k', type=int, default=64, help='transformer模型超参数:d_k')
    parser.add_argument('--d_v', type=int, default=64, help='transformer模型超参数:d_v')
    parser.add_argument('--d_ff', type=int, default=2048, help='transformer模型超参数:d_ff')
    parser.add_argument('--dropout', type=int, default=10, help='transformer模型超参数:dropout')
    params = parser.parse_args()

    params.temp_pt_file = os.path.join(params.main_data_dir, params.dataset_dir, params.temp_pt_file)
    params.checkpoint_dir = os.path.join(params.main_checkpoint_dir, params.dataset_dir)
    params.checkpoint_file = os.path.join(params.main_checkpoint_dir, params.dataset_dir, params.checkpoint_file)
    params.output_dir = os.path.join(params.main_output_dir, params.dataset_dir)
    params.pred_file = os.path.join(params.main_output_dir, params.dataset_dir, params.pred_file)

    # 原始的真实输出文件位置,需要从数据目录移动到输出目录下
    params.origin_gold_file = os.path.join(params.main_data_dir, params.dataset_dir, 'dev/question.txt')
    params.gold_file = os.path.join(params.main_output_dir, params.dataset_dir, params.gold_file)

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader, dev_loader = prepare_dataloaders(data, params)

    # 训练模型
    train_model(train_loader, dev_loader, vocab, params)
