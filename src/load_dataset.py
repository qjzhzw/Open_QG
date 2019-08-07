#!/usr/bin/env python
# encoding: utf-8
'''
加载数据集:
(1)将json形式的SQuAD原始文件(由Song et al., 2018提供)转换为txt形式,便于transformer模型进行处理
'''
__author__ = 'qjzhzw'

import json
import os
import random

from logger import logger
from params import params


def load_dataset(params, origin_file, sentence_file, question_file, answer_start_file, answer_end_file):
    '''
    作用:
    将json形式的原始文件中的[句子/问题/答案]三元组转换为txt形式

    输入参数:
    params: 参数集合
    origin_file: 输入的原始文件
    sentence_file: 输出的句子文件
    question_file: 输出的问题文件
    answer_start_file: 输出的答案开始位置文件
    answer_end_file: 输出的答案结束位置文件
    '''

    # 将原始数据加载进来
    instances = json.loads(open(origin_file, 'r').read())

    # total和num用于判断有多少数据被成功处理
    total = len(instances)
    num = 0

    # 所有输出文件
    f_sentence = open(sentence_file, 'w')
    f_question = open(question_file, 'w')
    f_answer_start = open(answer_start_file, 'w')
    f_answer_end = open(answer_end_file, 'w')

    # 依次处理所有数据
    for instance in instances:
        # 加载原始文件中的[句子/问题/答案]三元组
        sentence = instance['annotation1']['toks'].strip().lower()
        question = instance['annotation2']['toks'].strip().lower()
        answer = instance['annotation3']['toks'].strip().lower()

        # 将str切分为list
        sentence = sentence.split()
        question = question.split()
        answer = answer.split()

        # 找到答案起止位置
        answer_start = 0
        answer_end = 0
        for index in range(len(sentence)):
            if answer == sentence[index : index + len(answer)]:
                answer_start = index
                answer_end = index + len(answer)
                break

        # 将句子和答案连接起来(类似于bert中的做法)
        # <cls> 句子 <sep> 答案 <sep>
        if params.with_answer:
            sentence.insert(0, '<cls>')
            sentence.append('<sep>')
            sentence.extend(answer)
            sentence.append('<sep>')

        # 将处理好的数据存入本地文件
        # 如果得到的答案为空,则放弃该条数据
        if params.full_data or (answer_start != 0 or answer_end != 0):
            f_sentence.write(' '.join(sentence) + '\n')
            f_question.write(' '.join(question) + '\n')
            f_answer_start.write(str(answer_start) + '\n')
            f_answer_end.write(str(answer_end) + '\n')
            num += 1

    # 关闭所有文件
    f_sentence.close()
    f_question.close()
    f_answer_start.close()
    f_answer_end.close()

    logger.info('从{}中加载原始数据{}, 其中成功加载数据{}'.format(origin_file, total, num))


def load_dataset_translation(params, origin_sentence_file, origin_question_file,
                             train_sentence_file, train_question_file,
                             dev_sentence_file, dev_question_file,
                             test_sentence_file, test_question_file):
    '''
    作用:
    将txt形式的原始文件中进行划分得到训练集/验证集/测试集

    输入参数:
    params: 参数集合
    origin_sentence_file: 输入的句子文件
    origin_question_file: 输入的问题文件
    train_sentence_file: 输出的训练集的句子文件
    train_question_file: 输出的训练集的问题文件
    dev_sentence_file: 输出的验证集的句子文件
    dev_question_file: 输出的验证集的问题文件
    test_sentence_file: 输出的测试集的句子文件
    test_question_file: 输出的测试集的问题文件
    '''

    # 将原始数据加载进来
    sentences = open(origin_sentence_file, 'r').readlines()
    questions = open(origin_question_file, 'r').readlines()

    # 打乱原始数据(耗时间)
    zip_instances = list(zip(sentences, questions))
    random.shuffle(zip_instances)
    sentences, questions = zip(*zip_instances)

    # 断言: 句子和问题数量一致
    assert len(sentences) == len(questions)
    total = len(sentences)

    logger.info('总数据量{}'.format(total))

    # 所有输出文件
    f_train_sentence_file = open(train_sentence_file, 'w')
    f_train_question_file = open(train_question_file, 'w')
    f_dev_sentence_file = open(dev_sentence_file, 'w')
    f_dev_question_file = open(dev_question_file, 'w')
    f_test_sentence_file = open(test_sentence_file, 'w')
    f_test_question_file = open(test_question_file, 'w')

    # 划分训练集/验证集/测试集
    sentences_train = sentences[ : int(total * 0.007 * 5)]
    questions_train = questions[ : int(total * 0.007 * 5)]
    sentences_dev = sentences[int(total * 0.007 * 5) : int(total * 0.008 * 5)]
    questions_dev = questions[int(total * 0.007 * 5) : int(total * 0.008 * 5)]
    sentences_test = sentences[int(total * 0.008 * 5) : int(total * 0.010 * 5)]
    questions_test = questions[int(total * 0.008 * 5) : int(total * 0.010 * 5)]

    # 存入本地文件
    for sentence in sentences_train:
        f_train_sentence_file.write(sentence)
    for sentence in questions_train:
        f_train_question_file.write(sentence)
    for sentence in sentences_dev:
        f_dev_sentence_file.write(sentence)
    for sentence in questions_dev:
        f_dev_question_file.write(sentence)
    for sentence in sentences_test:
        f_test_sentence_file.write(sentence)
    for sentence in questions_test:
        f_test_question_file.write(sentence)

    logger.info('训练集{}'.format(len(sentences_train)))
    logger.info('验证集{}'.format(len(sentences_dev)))
    logger.info('测试集{}'.format(len(sentences_test)))


if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # 判断子目录train/dev/test是否存在，若不存在则创建
    if not os.path.exists(params.train_dir):
        os.makedirs(params.train_dir)
    if not os.path.exists(params.dev_dir):
        os.makedirs(params.dev_dir)
    if not os.path.exists(params.test_dir):
        os.makedirs(params.test_dir)

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 加载数据集
    if params.dataset_dir == 'translation':
        load_dataset_translation(params,
                            params.origin_sentence_file,
                            params.origin_question_file,
                            params.train_sentence_file,
                            params.train_question_file,
                            params.dev_sentence_file,
                            params.dev_question_file,
                            params.test_sentence_file,
                            params.test_question_file)
    else:
        load_dataset(params,
                    params.origin_train_file,
                    params.train_sentence_file,
                    params.train_question_file,
                    params.train_answer_start_file,
                    params.train_answer_end_file),
        load_dataset(params,
                    params.origin_dev_file,
                    params.dev_sentence_file,
                    params.dev_question_file,
                    params.dev_answer_start_file,
                    params.dev_answer_end_file)
        load_dataset(params,
                    params.origin_test_file,
                    params.test_sentence_file,
                    params.test_question_file,
                    params.test_answer_start_file,
                    params.test_answer_end_file)
