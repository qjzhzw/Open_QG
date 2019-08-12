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

        # encoder和decoder中部分参数共享
        if self.params.share_embeddings:
            self.decoder.word_embedding_decoder = self.encoder.word_embedding_encoder
            self.decoder.position_embedding_decoder = self.encoder.position_embedding_encoder
            self.decoder.output.weight = self.decoder.word_embedding_decoder.weight

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

        # 构造掩膜和位置信息
        self.utils = Utils(self.params)

        # embedding层,将索引/位置信息转换为词向量
        self.word_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.position_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.answer_embedding_encoder = nn.Embedding(2, self.params.d_model)

        # 如果有预训练的词向量,则使用预训练的词向量进行权重初始化
        if self.params.load_embeddings:
            weights = self.utils.init_embeddings(self.vocab)
            self.word_embedding_encoder = nn.Embedding.from_pretrained(weights, freeze=self.params.train_embeddings)

        # 多个相同子结构组成的encoder子层,层数为num_layers
        self.encoder_layers = nn.ModuleList([Encoder_layer(self.params) for _ in range(self.params.num_layers)])

    def forward(self, input_indices, answer_indices=None):
        '''
        输入参数:
        input_indices: [batch_size, input_seq_len]
        answer_indices: [batch_size, input_seq_len]

        输出参数:
        input_indices: [batch_size, input_seq_len]
        '''

        # 构造掩膜和位置信息
        input_indices_positions = self.utils.build_positions(input_indices)
        encoder_self_attention_masks = self.utils.build_pad_masks(query=input_indices, key=input_indices)
        # input_indices_positions: [batch_size, input_seq_len]
        # encoder_self_attention_masks: [batch_size, input_seq_len, input_seq_len]

        # 将索引/位置信息转换为词向量
        input_indices = self.word_embedding_encoder(input_indices) * np.sqrt(self.params.d_model) + \
                        self.position_embedding_encoder(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        # 如果有答案信息,就转换为词向量
        if torch.is_tensor(answer_indices):
            input_indices += self.answer_embedding_encoder(answer_indices)

        # 经过多个相同子结构组成的decoder子层,层数为num_layers
        for encoder_layer in self.encoder_layers:
            input_indices = encoder_layer(input_indices,
                                          encoder_self_attention_masks)
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

        # 构造掩膜和位置信息
        self.utils = Utils(self.params)

        # embedding层,将索引/位置信息转换为词向量
        self.word_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.position_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)

        # 如果有预训练的词向量,则使用预训练的词向量进行权重初始化
        if self.params.load_embeddings:
            weights = self.utils.init_embeddings(self.vocab)
            self.word_embedding_decoder = nn.Embedding.from_pretrained(weights, freeze=self.params.train_embeddings)

        # 多个相同子结构组成的decoder子层,层数为num_layers
        self.decoder_layers = nn.ModuleList([Decoder_layer(self.params) for _ in range(self.params.num_layers)])

        # 输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        self.output = nn.Linear(self.params.d_model, self.vocab_size)

        # copy机制所用的门控
        self.copy_gate = nn.Linear(self.params.d_model, 1)

    def forward(self, output_indices, input_indices, encoder_hiddens):
        '''
        输入参数:
        output_indices: [batch_size, output_seq_len]
        input_indices: [batch_size, input_seq_len]
        encoder_hiddens: [batch_size, input_seq_len, d_model]

        输出参数:
        output_indices: [batch_size, output_seq_len, vocab_size]
        '''

        # 构造掩膜和位置信息
        output_indices_positions = self.utils.build_positions(output_indices)
        decoder_mutual_attention_masks = self.utils.build_pad_masks(query=output_indices, key=input_indices)
        decoder_self_attention_masks = (self.utils.build_pad_masks(query=output_indices, key=output_indices) * \
                                        self.utils.build_triu_masks(output_indices)).gt(0)
        # output_indices_positions: [batch_size, output_seq_len]
        # decoder_mutual_attention_masks: [batch_size, output_seq_len, input_seq_len]
        # decoder_self_attention_masks: [batch_size, output_seq_len, output_seq_len]

        # 将索引/位置信息转换为词向量
        output_indices = self.word_embedding_decoder(output_indices) * np.sqrt(self.params.d_model) + \
                         self.position_embedding_decoder(output_indices)
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过多个相同子结构组成的decoder子层,层数为num_layers
        for decoder_layer in self.decoder_layers:
            output_indices, attention, context_vector = \
                decoder_layer(output_indices,
                              encoder_hiddens,
                              decoder_self_attention_masks,
                              decoder_mutual_attention_masks,
                              self.vocab)
        # output_indices: [batch_size, output_seq_len, d_model]
        # attention: [batch_size, output_seq_len, input_seq_len]

        if self.params.with_copy:
            copy_gate = self.copy_gate(output_indices)
            copy_gate = torch.sigmoid(copy_gate)
            # copy_gate: [batch_size, output_seq_len, 1]
            # print(copy_gate[-1].squeeze())

        # 经过输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        output_indices = self.output(output_indices)
        # output_indices: [batch_size, output_seq_len, vocab_size]

        # 使用softmax将模型输出转换为概率分布
        output_indices = F.softmax(output_indices, dim=-1)
        # # 将softmax后的结果控制在一定范围内,避免在计算log时出现log(0)的情况
        # output_indices = torch.clamp(output_indices, 1e-30, 1)

        if self.params.with_copy:
            # copy机制
            copy_indices = self.copy(attention, input_indices, output_indices)
            # copy_indices: [batch_size, output_seq_len, vocab_size]

            output_indices = copy_gate * copy_indices + (1 - copy_gate) * output_indices
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


class Encoder_layer(nn.Module):
    def __init__(self, params):
        '''
        Encoder_layer类:
        由于encoder部分每个子结构是相同的,因此单独抽象出来

        输入参数:
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        # self_attention结构
        self.self_attention = Multihead_attention(self.params)

        # FFN结构
        self.feedforward_network = Feedforward_network(self.params)

    def forward(self, input_indices, encoder_self_attention_masks):
        '''
        输入参数:
        input_indices: [batch_size, input_seq_len, d_model]
        encoder_self_attention_masks: [batch_size, input_seq_len, input_seq_len]

        输出参数:
        input_indices: [batch_size, input_seq_len, d_model]
        '''

        # 经过self_attention结构
        input_indices, _, _ = self.self_attention(query=input_indices,
                                                  key=input_indices,
                                                  value=input_indices,
                                                  mask=encoder_self_attention_masks)
        # input_indices: [batch_size, input_seq_len, d_model]

        # 经过FFN结构
        input_indices = self.feedforward_network(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        return input_indices


class Decoder_layer(nn.Module):
    def __init__(self, params):
        '''
        Decoder_layer类:
        由于decoder部分每个子结构是相同的,因此单独抽象出来

        输入参数:
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        # self_attention结构
        self.self_attention = Multihead_attention(self.params)

        # mutual_attention结构
        self.mutual_attention = Multihead_attention(self.params)

        # FFN结构
        self.feedforward_network = Feedforward_network(self.params)

    def forward(self, output_indices, encoder_hiddens, decoder_self_attention_masks, decoder_mutual_attention_masks, vocab):
        '''
        输入参数:
        output_indices: [batch_size, output_seq_len, d_model]
        encoder_hiddens: [batch_size, input_seq_len, d_model]
        decoder_self_attention_masks: [batch_size, output_seq_len, output_seq_len]
        decoder_mutual_attention_masks: [batch_size, output_seq_len, input_seq_len]
        vocab: Vocab类

        输出参数:
        output_indices: [batch_size, output_seq_len, d_model]
        '''

        # 经过self_attention结构
        output_indices, _, _ = self.self_attention(query=output_indices,
                                                   key=output_indices,
                                                   value=output_indices,
                                                   mask=decoder_self_attention_masks)
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过mutual_attention结构
        output_indices, attention, context_vector = \
            self.mutual_attention(query=output_indices,
                                  key=encoder_hiddens,
                                  value=encoder_hiddens,
                                  mask=decoder_mutual_attention_masks)
        # output_indices: [batch_size, output_seq_len, d_model]
        # attention: [batch_size, output_seq_len, input_seq_len]

        # 经过FFN结构
        output_indices = self.feedforward_network(output_indices)
        # output_indices: [batch_size, output_seq_len, d_model]

        return output_indices, attention, context_vector


class Multihead_attention(nn.Module):
    def __init__(self, params):
        '''
        Multihead_attention类:
        多头注意力机制,self-att和mutual-att都使用该类

        输入参数:
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        # 将输入通过三个各自的线性变换,转换为query/key/value
        self.linear_query = nn.Linear(self.params.d_model, self.params.d_k * self.params.num_heads)
        self.linear_key = nn.Linear(self.params.d_model, self.params.d_k * self.params.num_heads)
        self.linear_value = nn.Linear(self.params.d_model, self.params.d_v * self.params.num_heads)

        # nn.init.normal_(self.linear_query.weight, mean=0, std=np.sqrt(2.0 / (self.params.d_model + self.params.d_k)))
        # nn.init.normal_(self.linear_key.weight, mean=0, std=np.sqrt(2.0 / (self.params.d_model + self.params.d_k)))
        # nn.init.normal_(self.linear_value.weight, mean=0, std=np.sqrt(2.0 / (self.params.d_model + self.params.d_v)))

        # 将多头信息合并后通过线性变换
        self.linear_o = nn.Linear(self.params.num_heads * self.params.d_v, self.params.d_model)

        # dropout和层正则化
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.d_model)

    def forward(self, query, key, value, mask=None):
        '''
        输入参数:
        query: [batch_size, seq_len_query, d_model]
        key: [batch_size, seq_len_key, d_model]
        value: [batch_size, seq_len_key, d_model]
        mask: [batch_size, seq_len_query, seq_len_key]

        输出参数:
        indices: [batch_size, seq_len, d_model]
        '''
        # 保留进入子层时的输入状态,用于离开子层时的残差结构
        residual = query

        # 断言: query/key/value的batch_size和seq_len均一致
        assert query.size(0) == key.size(0) == value.size(0)
        assert key.size(1) == value.size(1)

        batch_size = query.size(0)
        seq_len_query = query.size(1)
        seq_len_key = key.size(1)

        # 将输入通过三个各自的线性变换,转换为query/key/value
        query = self.linear_query(query).view(batch_size, seq_len_query, self.params.num_heads, self.params.d_k)
        key = self.linear_key(key).view(batch_size, seq_len_key, self.params.num_heads, self.params.d_k)
        value = self.linear_value(value).view(batch_size, seq_len_key, self.params.num_heads, self.params.d_v)
        # query: [batch_size, seq_len_query, num_heads, d_k]
        # key: [batch_size, seq_len_key, num_heads, d_k]
        # value: [batch_size, seq_len_key, num_heads, d_v]

        # 将query/key/value转换维度,用于后面attention的计算
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # query: [batch_size, num_heads, seq_len_query, d_k]
        # key: [batch_size, num_heads, d_k, seq_len_key]
        # value: [batch_size, num_heads, seq_len_key, d_v]

        # 计算query和key的相似度,用于attention的计算
        attention_score = torch.matmul(query, key)
        # attention_score: [batch_size, num_heads, seq_len_query, seq_len_key]
        scaled_attention_score = attention_score / np.sqrt(self.params.d_k)
        # attention_score: [batch_size, num_heads, seq_len_query, seq_len_key]

        # 在计算attention分数时需要考虑mask,对于mask为0的部分用负无穷进行替代
        # mask: [batch_size, seq_len_query, seq_len_key]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.params.num_heads, 1, 1)
            # mask: [batch_size, num_heads, seq_len_query, seq_len_key]
            scaled_attention_score = scaled_attention_score.masked_fill(mask==0, -1e18)
        # scaled_attention_score: [batch_size, num_heads, seq_len_query, seq_len_key]

        # 通过softmax归一化后得到最终的attention分数,权重加和后得到context_vector向量
        attention = F.softmax(scaled_attention_score, dim=-1)
        # attention: [batch_size, num_heads, seq_len_query, seq_len_key]
        context_vector = torch.matmul(attention, value)
        # context_vetor: [batch_size, num_heads, seq_len_query, d_v]

        # 通过线性变换将多头信息合并
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_query, -1)
        # context_vector: [batch_size, seq_len_query, num_heads * d_v]
        indices = self.linear_o(context_vector)
        # indices: [batch_size, seq_len_query, d_model]

        # 将attention的多头信息合并,便于进行copy机制
        attention = torch.mean(attention, dim=1)
        # attention: [batch_size, seq_len_query, seq_len_key]

        # dropout
        indices = self.dropout(indices)
        # 残差结构
        indices += residual
        # 层正则化
        indices = self.layer_norm(indices)
        # indices: [batch_size, seq_len_query, d_model]

        return indices, attention, context_vector


class Feedforward_network(nn.Module):
    def __init__(self, params):
        '''
        Feedforward_network类:
        前馈网络(可用两个线性函数或是CNN实现)

        输入参数:
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        # 两个线性函数,中间用relu激活函数
        self.linear_FFN_1 = nn.Linear(self.params.d_model, self.params.d_ff)
        self.relu_between = nn.ReLU()
        self.linear_FFN_2 = nn.Linear(self.params.d_ff, self.params.d_model)

        # dropout和层正则化
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.d_model)

    def forward(self, indices):
        '''
        输入参数:
        indices: [batch_size, seq_len, d_model]

        输出参数:
        indices: [batch_size, seq_len, d_model]
        '''
        # 保留进入子层时的输入状态,用于离开子层时的残差结构
        residual = indices

        # 两个FFN,维度从d_model到d_ff再回到d_model
        indices = self.linear_FFN_1(indices)
        # indices: [batch_size, seq_len, d_ff]
        indices = self.relu_between(indices)
        # indices: [batch_size, seq_len, d_ff]
        indices = self.linear_FFN_2(indices)
        # indices: [batch_size, seq_len, d_model]

        # dropout
        indices = self.dropout(indices)
        # 残差结构
        indices += residual
        # 层正则化
        indices = self.layer_norm(indices)
        # indices: [batch_size, seq_len, d_model]

        return indices


class Utils():
    def __init__(self, params):
        '''
        Utils类:
        构造掩膜和位置信息

        输入参数:
        params: 参数集合
        '''
        self.params = params

    def build_positions(self, indices):
        '''
        作用:
        构造位置信息

        输入参数:
        indices: [batch_size, seq_len]

        输出参数:
        indices_positions: [batch_size, seq_len]
        '''

        # 构造位置信息,位置从0开始计算
        indices_positions = []
        for indice in indices:
            indice_position = []
            for position, index in enumerate(indice):
                indice_position.append(position)
            indices_positions.append(indice_position)

        # 将二维list的batch直接转换为tensor的形式
        indices_positions = torch.tensor(indices_positions).float().to(self.params.device)
        # indices_positions: [batch_size, seq_len]

        return indices_positions

    def build_pad_masks(self, query, key):
        '''
        作用:
        构造pad_mask(用于所有需要计算attention的地方,目的是过滤不为序列的部分即<pad>)

        输入参数:
        query: [batch_size, seq_len_query]
        key: [batch_size, seq_len_key]

        输出参数:
        indices_masks: [batch_size, seq_len_query, seq_len_key]
        '''

        batch_size = query.size(0)
        seq_len_query = query.size(1)
        seq_len_key = key.size(1)

        # 构造一个全为1的tensor
        indices_ones_masks = torch.ones_like(key).float().to(self.params.device)
        # indices_ones_masks: [batch_size, seq_len_key]

        # 是序列的部分赋值为0,不是序列的部分赋值为1
        indices_masks = key.eq(0).float()
        # indices_masks: [batch_size, seq_len_key]

        # 是序列的部分赋值为1,不是序列的部分赋值为0
        indices_masks = indices_ones_masks - indices_masks
        # indices_masks: [batch_size, seq_len_key]

        # 增加seq_len_query的维度
        indices_masks = indices_masks.unsqueeze(1).expand(-1, seq_len_query, -1)
        # indices_masks: [batch_size, seq_len_query, seq_len_key]

        return indices_masks

    def build_triu_masks(self, indices):
        '''
        作用:
        构造triu_mask(用于decoder的self-att,目的是防止decoder能够看到未来的信息)

        输入参数:
        indices: [batch_size, seq_len]

        输出参数:
        indices_triu_masks: [batch_size, seq_len, seq_len]
        '''

        batch_size = indices.size(0)
        seq_len = indices.size(1)

        # 构造一个全为1的tensor
        indices_ones_masks = torch.ones(seq_len, seq_len).to(self.params.device)
        # [seq_len, seq_len]

        # 构造成上三角矩阵,即右上均为1
        indices_triu_masks = torch.triu(indices_ones_masks, diagonal=1)
        # [seq_len, seq_len]

        # 构造成下三角矩阵,即左下均为1
        indices_triu_masks = indices_ones_masks - indices_triu_masks
        # [seq_len, seq_len]

        # 重复batch_size次
        indices_triu_masks = indices_triu_masks.unsqueeze(0).repeat(batch_size, 1, 1)
        # [batch_size, seq_len, seq_len]

        return indices_triu_masks

    def init_embeddings(self, vocab):
        '''
        作用:
        加载预训练的词向量权重

        输入参数:
        vocab: Vocab类

        输出参数:
        weights: [vocab_size, d_model]
        '''

        # 从Vocab类中按索引读取每个词的词向量
        weights = []
        for element in vocab.vocab:
            weights.append(element.embedding)

        # 将二维list的batch直接转换为tensor的形式
        weights = torch.tensor(weights).to(self.params.device)

        return weights
