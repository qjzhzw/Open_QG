### 数据预处理

import argparse
import logging
import json
import os
import torch

from vocab import Vocab

logger = logging.getLogger()


def load_dataset(origin_file):
    '''
    origin_file: 输入的原始文件
    return sentences: 输入文件的每个句子切分过后每个token组成二维list
    '''

    logger.info('正在从{}中加载数据'.format(origin_file))

    # 将原始数据加载进来
    instances = open(origin_file, 'r').readlines()

    sentences = []

    # 依次处理所有数据
    for instance in instances:
        words = instance.strip().split()
        sentence = ['<s>'] + words + ['</s>']
        sentences.append(sentence)
    
    logger.info('成功加载数据{}'.format(len(sentences)))

    return sentences


def load_answer(answer_start_file, answer_end_file):
    '''
    answer_start_file: 输出的答案开始位置文件
    answer_end_file: 输出的答案开始位置文件
    return answers: 每个答案开始/结束位置组成二维list
    '''

    logger.info('正在从{}和{}中加载数据'.format(answer_start_file, answer_end_file))

    # 将原始数据加载进来
    answer_starts = open(answer_start_file, 'r').readlines()
    answer_ends = open(answer_end_file, 'r').readlines()

    # 需要保证[答案开始位置/答案结束位置]数量一致
    assert len(answer_starts) == len(answer_ends)
    len_answer = len(answer_starts)

    answers = []

    # 依次处理所有数据
    for i in range(len_answer):
        answer_start = int(answer_starts[i].strip())
        answer_end = int(answer_ends[i].strip())
        answer = [answer_start, answer_end]
        answers.append(answer)
    
    logger.info('成功加载数据{}'.format(len(answers)))

    return answers


def build_vocab(sentences, vocab_dir):
    '''
    sentences: 输入所有样本
    vocab_dir: 输出vocab文件位置
    return vocab: 输出Vocab类,其中包含了数据集中的所有单词
    '''

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
    f_vocab = open(vocab_dir, 'w')

    # 构造vocab
    vocab = Vocab()
    # vocab在建立时,就已经包含了一定数量的常数,因此根据数据集添加单词时并不是以0作为起始索引
    index = len(vocab)
    for item in word_freq:
        word = item[0]
        freq = item[1]
        # 如果单词不在vocab中,就添加进去,否则放弃掉
        if not vocab.has_word(word):
            vocab.add_element(word, index, freq)
            index += 1

    # 将构造的vocab中,每个元素的信息(单词/索引)输出到文件中
    for element in vocab.vocab:
        f_vocab.write('{} {} {}\n'.format(element.word, element.index, element.freq))
    logger.info('vocab大小为{}'.format(len(vocab)))

    return vocab


def convert_sentence2index(sentences, vocab):
    '''
    sentences: 输入所有样本,均为单词形式
    vocab: Vocab类,其中包含了数据集中的所有单词
    return sentences_index: 输出所有样本,均为索引形式
    '''

    # 将单词转化为索引形式
    sentences_index = []
    for sentence in sentences:
        sentence_index = vocab.convert_sentence2index(sentence)
        sentences_index.append(sentence_index)

    return sentences_index


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
    parser.add_argument('--train_input', type=str, default='train/sentence.txt', help='训练集输入')
    parser.add_argument('--train_output', type=str, default='train/question.txt', help='训练集输出')
    parser.add_argument('--train_answer_start', type=str, default='train/answer_start.txt', help='训练集答案开始位置')
    parser.add_argument('--train_answer_end', type=str, default='train/answer_end.txt', help='训练集答案结束位置')
    parser.add_argument('--dev_input', type=str, default='dev/sentence.txt', help='验证集输入')
    parser.add_argument('--dev_output', type=str, default='dev/question.txt', help='验证集输出')
    parser.add_argument('--dev_answer_start', type=str, default='dev/answer_start.txt', help='验证集答案开始位置')
    parser.add_argument('--dev_answer_end', type=str, default='dev/answer_end.txt', help='验证集答案结束位置')
    parser.add_argument('--vocab_file', type=str, default='vocab.txt', help='vocab位置')
    parser.add_argument('--temp_pt_file', type=str, default='data.pt', help='输出的pt文件位置')
    params = parser.parse_args()

    params.train_input = os.path.join(params.main_data_dir, params.dataset_dir, params.train_input)
    params.train_output = os.path.join(params.main_data_dir, params.dataset_dir, params.train_output)
    params.train_answer_start = os.path.join(params.main_data_dir, params.dataset_dir, params.train_answer_start)
    params.train_answer_end = os.path.join(params.main_data_dir, params.dataset_dir, params.train_answer_end)
    params.dev_input = os.path.join(params.main_data_dir, params.dataset_dir, params.dev_input)
    params.dev_output = os.path.join(params.main_data_dir, params.dataset_dir, params.dev_output)
    params.dev_answer_start = os.path.join(params.main_data_dir, params.dataset_dir, params.dev_answer_start)
    params.dev_answer_end = os.path.join(params.main_data_dir, params.dataset_dir, params.dev_answer_end)
    params.vocab_file = os.path.join(params.main_data_dir, params.dataset_dir, params.vocab_file)
    params.temp_pt_file = os.path.join(params.main_data_dir, params.dataset_dir, params.temp_pt_file)

    # 将文件中的输出转化为二维list
    train_input_sentences = load_dataset(params.train_input)
    train_output_sentences = load_dataset(params.train_output)
    train_answers = load_answer(params.train_answer_start, params.train_answer_end)
    dev_input_sentences = load_dataset(params.dev_input)
    dev_output_sentences = load_dataset(params.dev_output)
    dev_answers = load_answer(params.dev_answer_start, params.dev_answer_end)

    # 需要保证[输入句子/输出句子/答案]数量一致
    assert len(train_input_sentences) == len(train_output_sentences) == len(train_answers)
    assert len(dev_input_sentences) == len(dev_output_sentences) == len(dev_answers)

    # 构造vocab
    vocab = build_vocab(train_input_sentences + train_output_sentences + dev_input_sentences + dev_output_sentences,
                        params.vocab_file)

    # 将单词转化为index
    train_input_indices = convert_sentence2index(train_input_sentences, vocab)
    train_output_indices = convert_sentence2index(train_output_sentences, vocab)
    dev_input_indices = convert_sentence2index(dev_input_sentences, vocab)
    dev_output_indices = convert_sentence2index(dev_output_sentences, vocab)

    # 构造数据,输出到临时的pt文件中
    data = {
        'vocab' : vocab,
        'train_input_indices' : train_input_indices,
        'train_output_indices' : train_output_indices,
        'dev_input_indices' : dev_input_indices,
        'dev_output_indices' : dev_output_indices,
    }
    torch.save(data, params.temp_pt_file)
