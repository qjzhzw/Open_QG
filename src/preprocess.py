#!/usr/bin/env python
# encoding: utf-8
'''
数据预处理:
(1)从txt形式的文件中读取模型需要使用的文本数据
(2)根据文本数据构造vocab
(3)根据vocab将数据从文本形式转换为索引形式
'''
__author__ = 'qjzhzw'

import json
import os
import random
import torch

from logger import logger
from params import params
from vocab import Vocab


def load_dataset(params, origin_file):
    '''
    作用:
    从txt形式的文件中读取模型需要使用的文本数据,转换为list形式

    输入参数:
    params: 参数集合
    origin_file: 输入的原始文件

    输出参数:
    sentences: 输入文件的每个句子切分过后每个token组成二维list
    '''

    # 将原始数据加载进来
    instances = open(origin_file, 'r').readlines()

    # 依次处理所有数据
    sentences = []
    for instance in instances:
        # 将str切分为list,过长的句子需要截短
        words = instance.strip().split()
        words = words[:params.max_seq_len]

        # 每个输入句子需要加入起止标志符<s>和</s>
        sentence = ['<s>'] + words + ['</s>']
        sentences.append(sentence)
    
    logger.info('从{}中成功加载数据{}'.format(origin_file, len(sentences)))

    return sentences


def load_answer(answer_start_file, answer_end_file):
    '''
    作用:
    从txt形式的文件中读取模型需要使用的文本数据(答案),转换为list形式

    输入参数:
    answer_start_file: 输出的答案开始位置文件
    answer_end_file: 输出的答案开始位置文件

    输出参数:
    answers: 每个答案开始/结束位置组成二维list
    '''

    # 将原始数据加载进来
    answer_starts = open(answer_start_file, 'r').readlines()
    answer_ends = open(answer_end_file, 'r').readlines()

    # 需要保证[答案开始位置/答案结束位置]数量一致
    assert len(answer_starts) == len(answer_ends)
    len_answer = len(answer_starts)

    # 依次处理所有数据
    answers = []
    for i in range(len_answer):
        # 读取答案开始和结束位置
        answer_start = int(answer_starts[i].strip())
        answer_end = int(answer_ends[i].strip())

        # 将[答案开始位置,答案结束位置]组成list
        answer = [answer_start, answer_end]
        answers.append(answer)
    
    logger.info('从{}和{}中成功加载数据{}'.format(answer_start_file, answer_end_file, len(answers)))

    return answers


def build_vocab(params, vocab_file, sentences):
    '''
    作用:
    根据文本数据构造vocab

    输入参数:
    params: 参数集合
    vocab_file: vocab文件所在位置
    sentences: 输入所有样本

    输出参数:
    vocab: 输出Vocab类,其中包含了数据集中的所有单词
    '''

    logger.info('正在构造vocab')

    # 统计词频
    word_freq = {}
    for sentence in sentences:
        for word in sentence:
            if word in word_freq.keys():
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    # 将word_freq按词频排序
    word_freq = sorted(word_freq.items(), key=lambda word: word[1], reverse=True)

    # 输出文件
    f_vocab = open(vocab_file, 'w')

    # 构造vocab
    vocab = Vocab(params)
    if vocab.word2embedding:
        logger.info('从{}中成功加载词向量{}'.format(params.embedding_file, len(vocab.word2embedding)))

    # 统计有多少单词被阈值过滤掉了,没有加入vocab
    dropped_word = 0

    # vocab在建立时,就已经包含了一定数量的常数,因此根据数据集添加单词时并不是以0作为起始索引
    index = len(vocab)
    for item in word_freq:
        word = item[0]
        freq = item[1]

        # 当词频高于一定阈值,才将该词加入vocab
        if freq >= params.min_word_count:
            # 如果单词有预训练的词向量,就将其作为该单词的词向量
            embedding = None
            if word in vocab.word2embedding.keys():
                embedding = vocab.word2embedding[word]

            # 如果单词不在vocab中,就添加进去,否则放弃掉
            if not vocab.has_word(word):
                vocab.add_element(word, index, freq, embedding)
                index += 1
        else:
            dropped_word += 1

    # 将构造的vocab中,每个元素的信息(单词/索引)输出到文件中
    for element in vocab.vocab:
        f_vocab.write('{} {} {} {}\n'.format(element.word, element.index, element.freq, element.embedding))

    logger.info('构造的vocab大小为{}'.format(len(vocab)))
    logger.info('被过滤的单词个数为{}'.format(dropped_word))

    return vocab


def load_vocab(params, vocab_file):
    '''
    作用:
    从预先设定好的vocab文件中加载vocab

    输入参数:
    params: 参数集合
    vocab_file: vocab文件所在位置

    输出参数:
    vocab: 输出Vocab类,其中包含了数据集中的所有单词
    '''

    logger.info('正在加载vocab')

    # 从已保存的vocab文件中读取vocab
    vocab_file = open(vocab_file, 'r')
    vocab = Vocab(params)

    # 逐行导入vocab元素
    for line in vocab_file:
        line = line.split()
        word = line[0]
        index = int(line[1])
        freq = line[2]
        embedding = line[3:]
        if not vocab.has_word(word):
            vocab.add_element(word, index, freq, embedding)

    logger.info('加载的vocab大小为{}'.format(len(vocab)))

    return vocab


def convert_sentence2index(sentences, vocab):
    '''
    作用:
    根据vocab将数据从文本形式转换为索引形式

    输入参数:
    sentences: 输入所有样本,均为文本形式
    vocab: Vocab类,其中包含了数据集中的所有单词

    输出参数:
    sentences_index: 输出所有样本,均为索引形式
    '''

    # 将文本形式转换为索引形式
    sentences_index = []
    for sentence in sentences:
        sentence_index = vocab.convert_sentence2index(sentence)
        sentences_index.append(sentence_index)

    return sentences_index


if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 将文件中的输出转换为二维list
    train_input_sentences = load_dataset(params, params.train_sentence_file)
    train_output_sentences = load_dataset(params, params.train_question_file)
    dev_input_sentences = load_dataset(params, params.dev_sentence_file)
    dev_output_sentences = load_dataset(params, params.dev_question_file)
    test_input_sentences = load_dataset(params, params.test_sentence_file)
    test_output_sentences = load_dataset(params, params.test_question_file)
    train_answers = dev_answers = test_answers = None
    if params.answer_embeddings:
        train_answers = load_answer(params.train_answer_start_file, params.train_answer_end_file)
        dev_answers = load_answer(params.dev_answer_start_file, params.dev_answer_end_file)
        test_answers = load_answer(params.test_answer_start_file, params.test_answer_end_file)

    # 断言:[句子/问题/答案]数量一致
    assert len(train_input_sentences) == len(train_output_sentences) 
    assert len(dev_input_sentences) == len(dev_output_sentences)
    assert len(test_input_sentences) == len(test_output_sentences)
    if params.answer_embeddings:
        assert len(train_input_sentences) == len(train_answers)
        assert len(dev_input_sentences) == len(dev_answers)
        assert len(test_input_sentences) == len(test_answers)

    # 加载/构造vocab
    if params.load_vocab and os.path.exists(params.vocab_file):
        vocab = load_vocab(params, params.vocab_file)
    else:
        vocab = build_vocab(params, params.vocab_file,
                            train_input_sentences + \
                            train_output_sentences + \
                            dev_input_sentences + \
                            dev_output_sentences)

    # 将单词转化为index
    train_input_indices = convert_sentence2index(train_input_sentences, vocab)
    train_output_indices = convert_sentence2index(train_output_sentences, vocab)
    dev_input_indices = convert_sentence2index(dev_input_sentences, vocab)
    dev_output_indices = convert_sentence2index(dev_output_sentences, vocab)
    test_input_indices = convert_sentence2index(test_input_sentences, vocab)
    test_output_indices = convert_sentence2index(test_output_sentences, vocab)

    logger.info('正在将数据中的单词转换为索引')

    # 构造数据,输出到临时的pt文件中
    data = {
        'params' : params,
        'vocab' : vocab,
        'train_input_indices' : train_input_indices,
        'train_output_indices' : train_output_indices,
        'train_answers' : train_answers,
        'dev_input_indices' : dev_input_indices,
        'dev_output_indices' : dev_output_indices,
        'dev_answers' : dev_answers,
        'test_input_indices' : test_input_indices,
        'test_output_indices' : test_output_indices,
        'test_answers' : test_answers,
    }
    torch.save(data, params.temp_pt_file)

    logger.info('构造数据输出已经保存至{}'.format(params.temp_pt_file))
