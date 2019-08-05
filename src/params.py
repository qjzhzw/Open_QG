#!/usr/bin/env python
# encoding: utf-8
'''
params方法:
(1)项目所用到的参数集合,所有文件共享
(2)如果某个python文件需要不同的参数,在对应文件中更改即可
'''
__author__ = 'qjzhzw'

import argparse
import os


def params():
    '''
    return params: 参数集合
    '''

    # 设定参数
    parser = argparse.ArgumentParser()

    # 文件夹位置相关
    parser.add_argument('--main_data_dir', type=str, default='data', help='数据主目录')
    parser.add_argument('--main_checkpoint_dir', type=str, default='checkpoint', help='输出的模型参数目录')
    parser.add_argument('--main_output_dir', type=str, default='output', help='输出的预测文件目录')
    parser.add_argument('--dataset_dir', type=str, default='translation', help='任务所使用的数据集所在的子目录')
    parser.add_argument('--origin_dir', type=str, default='origin', help='原始数据所在子目录')
    parser.add_argument('--train_dir', type=str, default='train', help='训练集数据所在子目录')
    parser.add_argument('--dev_dir', type=str, default='dev', help='验证集数据所在子目录')
    parser.add_argument('--test_dir', type=str, default='test', help='测试集数据所在子目录')

    # 文件位置相关
    parser.add_argument('--origin_train_file', type=str, default='train_sent_pre.json', help='原始训练集文件(json形式)')
    parser.add_argument('--origin_dev_file', type=str, default='dev_sent_pre.json', help='原始验证集文件(json形式)')
    parser.add_argument('--origin_test_file', type=str, default='test_sent_pre.json', help='原始测试集文件(json形式)')
    parser.add_argument('--sentence_file', type=str, default='sentence.txt', help='模型输入(句子)')
    parser.add_argument('--question_file', type=str, default='question.txt', help='模型输出(问题)')
    parser.add_argument('--answer_start_file', type=str, default='answer_start.txt', help='模型输入(答案开始位置)')
    parser.add_argument('--answer_end_file', type=str, default='answer_end.txt', help='模型输出(答案结束位置)')
    parser.add_argument('--embedding_file', type=str, default='embedding.txt', help='词向量的文件位置')
    parser.add_argument('--vocab_file', type=str, default='vocab.txt', help='vocab位置')
    parser.add_argument('--temp_pt_file', type=str, default='data.pt', help='暂存的pt文件位置')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pt', help='输出的模型参数位置')
    parser.add_argument('--pred_file', type=str, default='pred.txt', help='输出的预测文件位置')
    parser.add_argument('--gold_file', type=str, default='gold.txt', help='用于比较的真实文件位置')

    # 打印相关
    parser.add_argument('--print_params', type=bool, default=False, help='是否打印参数列表')
    parser.add_argument('--print_model', type=bool, default=False, help='是否打印出模型结构')
    parser.add_argument('--print_loss', type=bool, default=True, help='是否打印出训练过程中的损失')
    parser.add_argument('--print_results', type=bool, default=False, help='是否打印出训练过程中的预测序列')

    # 开关相关
    parser.add_argument('--with_answer', type=bool, default=False, help='是否在输入中加入答案信息')
    parser.add_argument('--full_data', type=bool, default=True, help='在没有找到答案信息的情况下是否保留该条数据')
    parser.add_argument('--max_seq_len', type=int, default=50, help='句子最大长度(多余的进行截短)')
    parser.add_argument('--load_vocab', type=bool, default=True, help='是否加载预先设定好的vocab')
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用cuda')
    parser.add_argument('--load_model', type=bool, default=True, help='是否加载训练好的模型参数')
    parser.add_argument('--label_smoothing', type=bool, default=True, help='是否使用标签平滑归一化')
    parser.add_argument('--load_embeddings', type=bool, default=False, help='是否加载预训练的词向量')
    parser.add_argument('--train_embeddings', type=bool, default=False, help='是否在训练过程中改变预训练的词向量')
    parser.add_argument('--with_copy', type=bool, default=False, help='是否使用copy机制')

    # 训练器超参数相关
    parser.add_argument('--num_epochs', type=int, default=1, help='模型超参数:num_epochs(模型训练/验证中设置)')
    parser.add_argument('--num_workers', type=int, default=0, help='模型超参数:num_workers(DataLoader中设置)')
    parser.add_argument('--batch_size', type=int, default=32, help='模型超参数:batch_size(批训练大小,DataLoader中设置)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='模型超参数:learning_rate(优化器中设置)')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='模型超参数:warmup_steps(优化器中设置)')
    parser.add_argument('--label_smoothing_eps', type=float, default=0.1, help='模型超参数:标签平滑归一化(计算损失时设置)')
    parser.add_argument('--beam_size', type=int, default=5, help='模型超参数:beam_size(模型测试中设置)')

    # 模型超参数相关
    parser.add_argument('--num_layers', type=int, default=3, help='transformer模型超参数:num_layers')
    parser.add_argument('--num_heads', type=int, default=3, help='transformer模型超参数:num_heads')
    parser.add_argument('--d_model', type=int, default=300, help='transformer模型超参数:d_model')
    parser.add_argument('--d_k', type=int, default=64, help='transformer模型超参数:d_k')
    parser.add_argument('--d_v', type=int, default=64, help='transformer模型超参数:d_v')
    parser.add_argument('--d_ff', type=int, default=1024, help='transformer模型超参数:d_ff')
    parser.add_argument('--dropout', type=float, default=0.1, help='transformer模型超参数:dropout')

    # 设定参数
    params = parser.parse_args()

    # 一些文件位置相关的参数需要调整
    params.origin_dir = os.path.join(params.main_data_dir, params.dataset_dir, params.origin_dir)
    params.origin_sentence_file = os.path.join(params.origin_dir, params.sentence_file)
    params.origin_question_file = os.path.join(params.origin_dir, params.question_file)

    params.origin_train_file = os.path.join(params.origin_dir, params.origin_train_file)
    params.train_dir = os.path.join(params.main_data_dir, params.dataset_dir, params.train_dir)
    params.train_sentence_file = os.path.join(params.train_dir, params.sentence_file)
    params.train_question_file = os.path.join(params.train_dir, params.question_file)
    params.train_answer_start_file = os.path.join(params.train_dir, params.answer_start_file)
    params.train_answer_end_file = os.path.join(params.train_dir, params.answer_end_file)

    params.origin_dev_file = os.path.join(params.origin_dir, params.origin_dev_file)
    params.dev_dir = os.path.join(params.main_data_dir, params.dataset_dir, params.dev_dir)
    params.dev_sentence_file = os.path.join(params.dev_dir, params.sentence_file)
    params.dev_question_file = os.path.join(params.dev_dir, params.question_file)
    params.dev_answer_start_file = os.path.join(params.dev_dir, params.answer_start_file)
    params.dev_answer_end_file = os.path.join(params.dev_dir, params.answer_end_file)

    params.origin_test_file = os.path.join(params.origin_dir, params.origin_test_file)
    params.test_dir = os.path.join(params.main_data_dir, params.dataset_dir, params.test_dir)
    params.test_sentence_file = os.path.join(params.test_dir, params.sentence_file)
    params.test_question_file = os.path.join(params.test_dir, params.question_file)
    params.test_answer_start_file = os.path.join(params.test_dir, params.answer_start_file)
    params.test_answer_end_file = os.path.join(params.test_dir, params.answer_end_file)

    params.embedding_file = os.path.join(params.origin_dir, params.embedding_file)
    params.vocab_file = os.path.join(params.main_data_dir, params.dataset_dir, params.vocab_file)
    params.temp_pt_file = os.path.join(params.main_data_dir, params.dataset_dir, params.temp_pt_file)

    params.checkpoint_dir = os.path.join(params.main_checkpoint_dir, params.dataset_dir)
    params.checkpoint_file = os.path.join(params.checkpoint_dir, params.checkpoint_file)

    params.output_dir = os.path.join(params.main_output_dir, params.dataset_dir)
    params.pred_file = os.path.join(params.output_dir, params.pred_file)
    params.gold_file = os.path.join(params.output_dir, params.gold_file)

    # 由于<s>和</s>的必然存在,因此真实的max_seq_len需要-2
    params.max_seq_len = params.max_seq_len - 2

    # 如果设置cuda且当前cuda可用,则设定为在cuda环境下运行
    try:
        import torch
        if params.cuda and torch.cuda.is_available():
            params.cuda = True
            params.device = torch.device('cuda')
        else:
            params.cuda = False
            params.device = torch.device('cpu')
    except:
        pass

    return params
