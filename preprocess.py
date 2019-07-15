### 数据预处理

import argparse
import logging
import json
import os

logger = logging.getLogger()

PAD = 0
UNK = 1
BOS = 2
EOS = 3
CLS = 4
SEP = 5

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
CLS_WORD = '<cls>'
SEP_WORD = '<sep>'

def load_dataset(origin_file):
    '''
    origin_file: 输入的原始文件
    sentences: 输入文件的每个句子切分过后每个token组成二维list
    '''

    logger.info('正在从{}中加载数据'.format(origin_file))

    # 将原始数据加载进来
    instances = open(origin_file, 'r').readlines()

    sentences = []

    # 依次处理所有数据
    for instance in instances:
        words = instance.strip().split()
        sentence = [BOS_WORD] + words + [EOS_WORD]
        sentences.append(sentence)
    
    logger.info('成功加载数据{}'.format(len(sentences)))

    return sentences


def load_answer(answer_start_file, answer_end_file):
    '''
    answer_start_file: 输出的答案开始位置文件
    answer_end_file: 输出的答案开始位置文件
    answers: 每个答案开始/结束位置组成二维list
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
    word_index = {}
    index = 0
    for item in word_freq:
        word = item[0]
        freq = item[1]
        word_index[word] = index
        index += 1
        f_vocab.write(word + ' ' + str(freq) + '\n')

    logger.info('vocab大小为{}'.format(len(word_index)))

    return word_index


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
    parser.add_argument('--train_input', type=str, default='train/sentence.txt', help='训练集输入')
    parser.add_argument('--train_output', type=str, default='train/question.txt', help='训练集输出')
    parser.add_argument('--train_answer_start', type=str, default='train/answer_start.txt', help='训练集答案开始位置')
    parser.add_argument('--train_answer_end', type=str, default='train/answer_end.txt', help='训练集答案结束位置')
    parser.add_argument('--dev_input', type=str, default='dev/sentence.txt', help='验证集输入')
    parser.add_argument('--dev_output', type=str, default='dev/question.txt', help='验证集输出')
    parser.add_argument('--dev_answer_start', type=str, default='dev/answer_start.txt', help='验证集答案开始位置')
    parser.add_argument('--dev_answer_end', type=str, default='dev/answer_end.txt', help='验证集答案结束位置')
    parser.add_argument('--vocab_dir', type=str, default='vocab.txt', help='vocab位置')
    params = parser.parse_args()

    train_input = os.path.join(params.main_data_dir, params.data_dir, params.train_input)
    train_output = os.path.join(params.main_data_dir, params.data_dir, params.train_output)
    train_answer_start = os.path.join(params.main_data_dir, params.data_dir, params.train_answer_start)
    train_answer_end = os.path.join(params.main_data_dir, params.data_dir, params.train_answer_end)
    dev_input = os.path.join(params.main_data_dir, params.data_dir, params.dev_input)
    dev_output = os.path.join(params.main_data_dir, params.data_dir, params.dev_output)
    dev_answer_start = os.path.join(params.main_data_dir, params.data_dir, params.dev_answer_start)
    dev_answer_end = os.path.join(params.main_data_dir, params.data_dir, params.dev_answer_end)
    vocab_dir = os.path.join(params.main_data_dir, params.data_dir, params.vocab_dir)

    # 将文件中的输出转化为二维list
    train_input_sentences = load_dataset(train_input)
    train_output_sentences = load_dataset(train_output)
    train_answers = load_answer(train_answer_start, train_answer_end)
    dev_input_sentences = load_dataset(dev_input)
    dev_output_sentences = load_dataset(dev_output)
    dev_answers = load_answer(dev_answer_start, dev_answer_end)

    # 需要保证[输入句子/输出句子/答案]数量一致
    assert len(train_input_sentences) == len(train_output_sentences) == len(train_answers)
    assert len(dev_input_sentences) == len(dev_output_sentences) == len(dev_answers)

    build_vocab(train_input_sentences + train_output_sentences + dev_input_sentences + dev_output_sentences, vocab_dir)
