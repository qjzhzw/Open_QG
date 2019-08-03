#!/usr/bin/env python
# encoding: utf-8
'''
Vocab类:
(1)核心是self.vocab，这是一个集合，包含了所有的元素，这些元素的类型为Vocab_element
(2)Vocab_element是Vocab类中的每个元素，包含word/index/freq/embedding
'''
__author__ = 'qjzhzw'

import random
import torch


class Vocab():
    def __init__(self, params):
        '''
        Vocab类:
        核心是self.vocab，这是一个集合，包含了所有的元素，这些元素的类型为Vocab_element

        输入参数:
        params: 参数集合
        '''

        self.params = params

        self.vocab = []
        self.word2index = {}
        self.index2word = {}
        self.word2embedding = {}

        # 从文件中加载所有单词的embedding
        self.embedding_size = self.params.d_model
        if self.params.load_embeddings:
            self.load_embeddings()
            # 断言: embedding_size和d_model相同
            assert self.embedding_size == self.params.d_model

        # 将常数加入到vocab中
        # <pad>: batch中由于数据长度要统一,因此多余部分均填<pad>
        # <unk>: 不在vocab中的词用<unk>替代
        # <s>: 序列开始符号
        # </s>: 序列结束符号
        # <cls>/<sep>: <cls> 句子 <sep> 答案 <sep>
        self.constants = ['<pad>', '<unk>', '<s>', '</s>', '<cls>', '<sep>']
        for index, constant in enumerate(self.constants):
            self.add_element(constant, index)

    def add_element(self, word, index, freq=None, embedding=None):
        '''
        作用:
        在Vocab类中加入一个元素
        必选项:单词,索引
        可选性:词频,词向量

        输入参数:
        word: 需要添加元素的单词
        index: 需要添加元素的索引
        freq: 需要添加元素的词频(可选)
        embedding: 需要添加元素的词向量(可选)
        '''

        # 如果传入的freq和embedding参数为空,进行初始化
        if freq == None:
            freq = 0
        if embedding == None:
            embedding = [random.uniform(-0.5, 0.5) for i in range(self.embedding_size)]

        # 在Vocab类中加入一个元素
        self.vocab.append(Vocab_element(word, index, freq, embedding))
        self.word2index[word] = index
        self.index2word[index] = word

    def __len__(self):
        '''
        作用:
        计算Vocab类的大小,即包含了多少[单词,索引]这样的元素

        输出参数:
        Vocab类的大小
        '''
        
        return len(self.vocab)

    def has_word(self, word):
        '''
        作用:
        判断Vocab类中是否包含某个单词

        输入参数:
        word: 输入单词

        输出参数:
        True: 表示单词在vocab中
        False: 表示单词不在vocab中
        '''

        if word in self.word2index.keys():
            return True
        else:
            return False

    def convert_word2index(self, word):
        '''
        作用:
        将一个单词转换为索引

        输入参数:
        word: 输入单词

        输出参数:
        index: 输出索引
        '''

        # 如果vocab中包含这个单词,则返回该单词的索引,否则返回<unk>的索引
        index = None
        if self.has_word(word):
            index = self.word2index[word]
        else:
            index = self.word2index['<unk>']
        return index

    def convert_index2word(self, index):
        '''
        作用:
        将一个索引转换为单词

        输入参数:
        index: 输入索引

        输出参数:
        word: 输出单词
        '''

        # 如果索引是合法的(即小于vocab的大小),则返回该索引的单词,否则返回<unk>
        word = None
        if index >= 0 and index < len(self):
            word = self.index2word[index]
        else:
            word = '<unk>'
        return word

    def convert_sentence2index(self, sentence):
        '''
        作用:
        将一个句子,从单词序列转换为索引形式

        输入参数:
        sentence: 输入单词序列

        输出参数:
        indices: 输出索引序列
        '''

        # 如果输入是tensor形式,转换为numpy形式
        # 如果数据在cuda上,必须先转到cpu上才可以转换
        if torch.is_tensor(sentence):
            sentence = sentence.cpu().numpy()

        # 通过遍历的方式,将单词序列转换为索引形式
        indices = []
        for word in sentence:
            indices.append(self.convert_word2index(word))
        return indices

    def convert_index2sentence(self, indices, mode=False):
        '''
        作用:
        将一个句子,从索引序列转换为单词形式

        输入参数:
        indices: 输入索引序列
        mode: True表示输出完整序列
              False表示遇到</s>就停止(只输出到</s>前的序列)

        输出参数:
        sentence: 输出单词序列
        '''

        # 如果输入是tensor形式,转换为numpy形式
        # 如果数据在cuda上,必须先转到cpu上才可以转换
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()

        # 通过遍历的方式,将索引序列转换为单词形式
        sentence = []
        for index in indices:
            # 在mode为False时,遇到</s>就停止
            if index == self.convert_word2index('</s>') and mode == False:
                break
            else:
                sentence.append(self.convert_index2word(index))
        return sentence

    def load_embeddings(self):
        '''
        作用:
        从文件中加载所有单词的词向量
        '''

        lines = open(self.params.embedding_file, 'r').readlines()
        for line in lines:
            line = line.split()

            # 每一行的结构: index(1维) word(1维) embedding(300维)
            word = line[1]
            embedding = line[-self.embedding_size:]

            # 将文件中的str类型转换为float类型
            embedding = [float(emb) for emb in embedding]
            self.word2embedding[word] = embedding
            self.embedding_size = len(embedding)


class Vocab_element():
    def __init__(self, word=None, index=None, freq=None, embedding=None):
        '''
        Vocab_element类:
        Vocab_element是Vocab类中的每个元素，包含word/index/freq/embedding
        '''
        
        self.word = word
        self.index = index
        self.freq = freq
        self.embedding = embedding
