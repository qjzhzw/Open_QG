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
from beam import Translator

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
        shuffle = False
    )
    logger.info('正在构造test_loader,共有{}个batch'.format(len(test_loader)))

    return test_loader


def test_model(test_loader, vocab, params):
    '''
    test_loader: 测试集的dataloader
    vocab: 从pt文件中读取的该任务所使用的vocab
    params: 参数集合
    '''

    logger.info('正在加载模型,即将开始测试')

    # 定义模型
    model = Model(params, vocab)

    # 如果参数中设置了使用cuda且当前cuda可用,则将模型放到cuda上
    flag_cuda = False
    if params.cuda:
        model.cuda()
        flag_cuda = True

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 断言: 模型参数文件存在
    assert os.path.exists(params.checkpoint_file)
    # 加载模型参数
    model_params = torch.load(params.checkpoint_file)
    model.load_state_dict(model_params)
    logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))

    # 一轮模型测试
    sentences_pred = one_epoch(model, test_loader, vocab, params)

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
    shutil.copyfile(params.origin_gold_file, params.gold_file)

    # 使用multi-bleu.perl的脚本对结果进行评估
    os.system('evaluate/multi-bleu.perl %s < %s' %(params.gold_file, params.pred_file))


def one_epoch(model, loader, vocab, params):
    '''
    model: 当前使用模型
    loader: 测试集的dataloader
    vocab: 从pt文件中读取的该任务所使用的vocab
    params: 参数集合
    return sentences_pred: 测试结束时得到的预测序列集合(只有验证时有内容,训练时为空list)
    '''

    logger.info('测试阶段')
    model.eval()

    # 我们保存所有得到的输出序列
    sentences_pred = []

    # 每一个batch的测试
    for batch_index, batch in enumerate(tqdm(loader)):
        # 从数据中读取模型的输入和输出
        input_indices = batch[0]
        output_indices = batch[1]

        # 如果参数中设置了使用cuda且当前cuda可用,则将数据放到cuda上
        if params.cuda:
            input_indices = input_indices.cuda()
            output_indices = output_indices.cuda()

        translator = Translator(params, model)
        all_hyp, all_scores = translator.translate_batch(input_indices)
        for i in all_hyp:
            sentence = vocab.convert_index2sentence(i[0])
            sentences_pred.append(' '.join(sentence))
        print(' '.join(vocab.convert_index2sentence(input_indices[-1])))
        print(sentences_pred[-1])

        # # # 起始符号均为<s>
        # batch_size = input_indices.size(0)
        # output_indices = torch.zeros(batch_size, 1).long().cuda()
        # for i, output_indice in enumerate(output_indices):
        #     output_indices[i] = vocab.convert_word2index('<s>')
        # # output_indices: [batch_size, output_seq_len(1)]

        # # ['<s>', 'what']
        # # output_indices = torch.tensor([[2]])

        # # 如果参数中设置了使用cuda且当前cuda可用,则将数据放到cuda上
        # if params.cuda:
        #     input_indices = input_indices.cuda()
        #     output_indices = output_indices.cuda()

        # encoder_hiddens = model.encoder(input_indices)
        # # encoder_hiddens: [batch_size, input_seq_len, d_model]

        # # output_seq_len的取值:1,2,3,4,5,……
        # for i in range(20):
        #     output_indices_pred = model.decoder(output_indices, input_indices, encoder_hiddens)
        #     # output_indices_pred: [batch_size, output_seq_len, vocab_size]

        #     # 取预测最大值作为输出
        #     indices_pred = torch.max(output_indices_pred, dim=-1)[1]
        #     # print(vocab.convert_index2sentence(indices_pred[0], mode=False))
        #     # indices_pred: [batch_size, output_seq_len]

        #     # 取模型输出的最后一个字符
        #     indices_pred = indices_pred[:, -1].unsqueeze(1)
        #     # indices_pred: [batch_size, output_seq_len(1)]

        #     output_indices = torch.cat((output_indices, indices_pred), dim=-1)
        #     # output_indices: [batch_size, output_seq_len + 1]
        
        # # 输出预测序列
        # for indices in output_indices:
        #     # mode: True表示输出完整序列
        #     #       False表示遇到</s>就停止(只输出到</s>前的序列)
        #     sentence = vocab.convert_index2sentence(indices, mode=False)
        #     sentence = sentence[1:]
        #     sentences_pred.append(' '.join(sentence))
        # print(sentence)

    return sentences_pred


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
    parser.add_argument('--print_model', type=bool, default=False, help='是否打印出模型结构')
    parser.add_argument('--print_results', type=bool, default=False, help='是否打印出训练过程中的预测序列')
    parser.add_argument('--num_workers', type=int, default=0, help='模型超参数:num_workers(DataLoader中设置)')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size(批训练大小,DataLoader中设置)')
    parser.add_argument('--embedding_size', type=int, default=512, help='transformer模型超参数:embedding_size(词向量维度,在transformer模型中和d_model一致)')
    parser.add_argument('--num_layers', type=int, default=6, help='transformer模型超参数:num_layers')
    parser.add_argument('--num_heads', type=int, default=8, help='transformer模型超参数:num_heads')
    parser.add_argument('--d_model', type=int, default=512, help='transformer模型超参数:d_model')
    parser.add_argument('--d_k', type=int, default=64, help='transformer模型超参数:d_k')
    parser.add_argument('--d_v', type=int, default=64, help='transformer模型超参数:d_v')
    parser.add_argument('--d_ff', type=int, default=2048, help='transformer模型超参数:d_ff')
    parser.add_argument('--dropout', type=float, default=0.1, help='transformer模型超参数:dropout')
    params = parser.parse_args()

    params.temp_pt_file = os.path.join(params.main_data_dir, params.dataset_dir, params.temp_pt_file)
    params.checkpoint_dir = os.path.join(params.main_checkpoint_dir, params.dataset_dir)
    params.checkpoint_file = os.path.join(params.main_checkpoint_dir, params.dataset_dir, params.checkpoint_file)
    params.output_dir = os.path.join(params.main_output_dir, params.dataset_dir)
    params.pred_file = os.path.join(params.main_output_dir, params.dataset_dir, params.pred_file)
    if params.cuda and torch.cuda.is_available():
        params.cuda = True
    else:
        params.cuda = False

    # 原始的真实输出文件位置,需要从数据目录移动到输出目录下
    params.origin_gold_file = os.path.join(params.main_data_dir, params.dataset_dir, 'test/question.txt')
    params.gold_file = os.path.join(params.main_output_dir, params.dataset_dir, params.gold_file)

    # 打印参数列表
    logger.info('参数列表:{}'.format(params))

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    test_loader = prepare_dataloaders(data, params)

    # 测试模型
    test_model(test_loader, vocab, params)
