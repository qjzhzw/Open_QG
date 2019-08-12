#!/usr/bin/env python
# encoding: utf-8
'''
Model类:
(1)Transformer模型,包含论文中提到的全部内容
'''
__author__ = 'qjzhzw'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, params, vocab):
        '''
        Model类:
        Transformer模型

        输入参数:
        params: 参数集合
        vocab: Vocab类,其中包含了数据集中的所有单词
        '''
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.encoder = Encoder(self.params, self.vocab)
        self.decoder = Decoder(self.params, self.vocab)

    def forward(self, input_indices, output_indices, answer_indices=None):
        '''
        输入参数:
        input_indices: [batch_size, input_seq_len]
        output_indices: [batch_size, output_seq_len]
        answer_indices: [batch_size, input_seq_len]

        输出参数:
        output_indices: [batch_size, output_seq_len, vocab_size]
        '''

        encoder_hiddens = self.encoder(input_indices, answer_indices)
        output_indices = self.decoder(output_indices, input_indices, encoder_hiddens)

        return output_indices


class Encoder(nn.Module):
    def __init__(self, params, vocab):
        '''
        Encoder类:
        模型的encoder部分

        输入参数:
        params: 参数集合
        vocab: Vocab类,其中包含了数据集中的所有单词
        '''
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # embedding层,将索引/位置信息转换为词向量
        self.word_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.answer_embedding_encoder = nn.Embedding(2, self.params.d_model)

        # 测试所使用GRU
        self.GRU_encoder = nn.GRU(self.params.d_model, self.params.d_model, batch_first=True, num_layers=1)

    def forward(self, input_indices, answer_indices=None):
        '''
        输入参数:
        input_indices: [batch_size, input_seq_len]
        answer_indices: [batch_size, input_seq_len]

        输出参数:
        input_indices: [batch_size, input_seq_len]
        '''

        # 测试所使用GRU
        input_indices = self.word_embedding_encoder(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        input_indices, _ = self.GRU_encoder(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        return input_indices


class Decoder(nn.Module):
    def __init__(self, params, vocab):
        '''
        Decoder类:
        模型的decoder部分

        输入参数:
        params: 参数集合
        vocab: Vocab类,其中包含了数据集中的所有单词
        '''
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # embedding层,将索引/位置信息转换为词向量
        self.word_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)

        # 测试所使用GRU
        self.GRU_decoder = nn.GRU(self.params.d_model, self.params.d_model, batch_first=True, num_layers=1)

        # 输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        self.output = nn.Linear(self.params.d_model, self.vocab_size)

    def forward(self, output_indices, input_indices, encoder_hiddens):
        '''
        输入参数:
        output_indices: [batch_size, output_seq_len]
        input_indices: [batch_size, input_seq_len]
        encoder_hiddens: [batch_size, input_seq_len, d_model]

        输出参数:
        output_indices: [batch_size, output_seq_len, vocab_size]
        '''

        # 测试所使用GRU
        output_indices = self.word_embedding_decoder(output_indices)
        # output_indices: [batch_size, output_seq_len, d_model]

        encoder_hiddens = encoder_hiddens[:, -1, :].unsqueeze(0).contiguous()
        # encoder_hiddens: [1, batch_size, d_model]

        output_indices, _ = self.GRU_decoder(output_indices, encoder_hiddens)
        # output_indices: [batch_size, output_seq_len, d_model]

        output_indices = self.output(output_indices)
        # output_indices: [batch_size, output_seq_len, vocab_size]

        output_indices = F.softmax(output_indices, dim=-1)
        # output_indices: [batch_size, output_seq_len, vocab_size]

        return output_indices

    def copy(self, attention, input_indices, output_indices):
        '''
        作用:
        copy机制

        输入:
        attention: [batch_size, output_seq_len, input_seq_len]
        input_indices: [batch_size, input_seq_len]
        output_indices: [batch_size, output_seq_len, vocab_size]

        输出:
        copy_indices: [batch_size, output_seq_len, vocab_size]
        '''

        batch_size = attention.size(0)
        output_seq_len = attention.size(1)
        input_seq_len = attention.size(2)

        # print(attention[-1])
        # att = torch.max(attention[-1], dim=-1)[1]
        # input_sen = self.vocab.convert_index2sentence(input_indices[-1])
        # print('注意力 : {}'.format([input_sen[i] for i in att]))

        copy_indices = torch.zeros_like(output_indices)
        # copy_indices: [batch_size, output_seq_len, vocab_size]
        input_indices = input_indices.unsqueeze(1).expand(-1, output_seq_len, -1)
        # input_indices: [batch_size, output_seq_len, input_seq_len]

        copy_indices = copy_indices.scatter_(2, input_indices, attention)
        # copy_indices: [batch_size, output_seq_len, vocab_size]

        # a = torch.max(copy_indices[-1], dim=-1)[1]
        # print(self.vocab.convert_index2sentence(a))

        return copy_indices
