#!/usr/bin/env python
# encoding: utf-8
'''
测试demo:
(1)人为输入句子和答案
(2)将句子和答案输入模型进行预测,得到预测问题
(3)输出预测问题
'''
__author__ = 'qjzhzw'

import os
import torch

from logger import logger
from params import params
from vocab import Vocab
from model import Model
from beam import Generator


def test_demo(params, vocab, input_sentence):
    '''
    作用:
    测试demo

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    input_sentence: 输入句子(文本形式)

    输出参数:
    output_sentence: 输出句子(文本形式)
    '''

    # 将文本转化为list,并添加起止符<s>和</s>
    input_sentence = input_sentence.split()
    input_sentence = ['<s>'] + input_sentence + ['</s>']

    # 将输入句子转化为索引形式
    input_indices = vocab.convert_sentence2index(input_sentence)
    logger.info('输入句子的索引形式为 : {}'.format(input_indices))

    # 定义模型
    model = Model(params, vocab).to(params.device)

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 断言: 模型参数文件存在
    assert os.path.exists(params.checkpoint_file)
    # 加载模型参数
    model_params = torch.load(params.checkpoint_file, map_location=params.device)
    model.load_state_dict(model_params)
    logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))

    logger.info('测试demo')
    model.eval()

    # 输入模型
    input_indices = torch.tensor(input_indices).to(params.device)
    # input_indices: [input_seq_len]
    input_indices = input_indices.unsqueeze(0)
    # input_indices: [1(batch_size), input_seq_len]

    # 使用beam_search算法,以<s>作为开始符得到完整的预测序列
    generator = Generator(params, model)
    indices_pred, scores_pred = generator.generate_batch(input_indices)
    # indices_pred: [batch_size, beam_size, output_seq_len]
    output_indices = indices_pred[0][0]
    # output_indices: [output_seq_len]

    logger.info('输出句子的索引形式为 : {}'.format(output_indices))

    # 将输出句子转化为文本形式
    output_sentence = vocab.convert_index2sentence(output_indices)
    output_sentence = ' '.join(output_sentence)

    return output_sentence


def demo(input_sentence, input_answer):
    '''
    作用:
    测试demo

    输入参数:
    input_sentence: 输入句子(文本形式)
    input_answer: 输入答案(文本形式)

    输出参数:
    output_question: 输出问题(文本形式)
    '''

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 从已保存的pt文件中读取vocab
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']

    input_sentence = '<cls> ' + input_sentence + ' <sep> ' + input_answer + ' <sep>'
    logger.info('输入句子的文本形式为 : {}'.format(input_sentence))

    # 测试demo
    output_question = test_demo(params, vocab, input_sentence)

    logger.info('输出问题的文本形式为 : {}'.format(output_question))

    return output_question


if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()
    params.max_seq_len = 30

    # 测试demo: 输入句子和答案,输出问题
    output_question = demo(input_sentence = 'There are 5000000 people in the united states .',
                    input_answer = '5000000')
