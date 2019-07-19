### 模型

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, params, vocab):
        '''
        params: 参数集合
        vocab: Vocab类,其中包含了数据集中的所有单词
        '''
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.encoder = Encoder(self.params, self.vocab)
        self.decoder = Decoder(self.params, self.vocab)

    def forward(self, input_indices, output_indices):
        '''
        input_indices: [batch_size, input_seq_len]
        output_indices: [batch_size, output_seq_len]
        return output_indices: [batch_size, vocab_size, output_seq_len]
        '''

        hidden = self.encoder(input_indices)
        output_indices = self.decoder(output_indices, hidden)

        return output_indices


class Encoder(nn.Module):
    def __init__(self, params, vocab):
        '''
        params: 参数集合
        vocab: Vocab类,其中包含了数据集中的所有单词
        '''
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # embedding层,将索引转换为词向量
        self.embedder = nn.Embedding(self.vocab_size, self.params.d_model)

        # self_attention结构
        self.self_attention = Multihead_attention(self.params)

        # FFN结构
        self.feedforward_network = Feedforward_network(self.params)

    def forward(self, input_indices):
        '''
        input_indices: [batch_size, input_seq_len]
        return input_indices: [batch_size, input_seq_len]
        '''

        # 将索引转换为词向量
        input_indices = self.embedder(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        # 经过self_attention结构
        input_indices = self.self_attention(input_indices, input_indices, input_indices)

        # 经过FFN结构
        input_indices = self.feedforward_network(input_indices)

        return input_indices


class Decoder(nn.Module):
    def __init__(self, params, vocab):
        '''
        params: 参数集合
        vocab: Vocab类,其中包含了数据集中的所有单词
        '''
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # embedding层,将索引转换为词向量
        self.embedder = nn.Embedding(self.vocab_size, self.params.d_model)

        # self_attention结构
        self.self_attention = Multihead_attention(self.params)

        # mutual_attention结构
        self.mutual_attention = Multihead_attention(self.params)

        # FFN结构
        self.feedforward_network = Feedforward_network(self.params)

        # 输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        self.output = nn.Linear(self.params.d_model, self.vocab_size)

    def forward(self, output_indices, hidden):
        '''
        output_indices: [batch_size, output_seq_len]
        return output_indices: [batch_size, vocab_size, output_seq_len]
        '''

        # 将索引转换为词向量
        output_indices = self.embedder(output_indices)
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过self_attention结构
        output_indices = self.self_attention(output_indices, output_indices, output_indices)

        # 经过mutual_attention结构
        output_indices = self.mutual_attention(output_indices, output_indices, output_indices)

        # 经过FFN结构
        output_indices = self.feedforward_network(output_indices)

        # 经过输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        output_indices = self.output(output_indices)
        # output_indices: [batch_size, output_seq_len, vocab_size]
        output_indices = output_indices.permute(0, 2, 1)
        # output_indices: [batch_size, vocab_size, output_seq_len]

        return output_indices


class Multihead_attention(nn.Module):
    def __init__(self, params):
        '''
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        self.linear_query = nn.Linear(self.params.d_model, self.params.d_k * self.params.num_heads)
        self.linear_key = nn.Linear(self.params.d_model, self.params.d_k * self.params.num_heads)
        self.linear_value = nn.Linear(self.params.d_model, self.params.d_v * self.params.num_heads)

        self.softmax_attention = nn.Softmax(dim=-1)

        self.linear_o = nn.Linear(self.params.num_heads * self.params.d_v, self.params.d_model)

    def forward(self, query, key, value):
        '''
        query: [batch_size, seq_len, d_model]
        key: [batch_size, seq_len, d_model]
        value: [batch_size, seq_len, d_model]
        return output: [batch_size, seq_len, d_model]
        '''
        # 断言: query/key/value的batch_size和seq_len均一致
        assert query.size(0) == key.size(0) == value.size(0)
        assert query.size(1) == key.size(1) == value.size(1)

        batch_size = query.size(0)
        seq_len = query.size(1)

        # 将输入通过三个各自的linear转换为query/key/value
        query = self.linear_query(query).view(batch_size, seq_len, self.params.num_heads, self.params.d_k)
        key = self.linear_key(key).view(batch_size, seq_len, self.params.num_heads, self.params.d_k)
        value = self.linear_value(value).view(batch_size, seq_len, self.params.num_heads, self.params.d_v)
        # query: [batch_size, seq_len, num_heads, d_k]
        # key: [batch_size, seq_len, num_heads, d_k]
        # value: [batch_size, seq_len, num_heads, d_v]

        # 将query/key/value转换维度,用于后面attention的计算
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # query: [batch_size, num_heads, seq_len, d_k]
        # key: [batch_size, num_heads, d_k, seq_len]
        # value: [batch_size, num_heads, seq_len, d_v]

        # attention的计算,通过attention分数得到context_vector向量
        attention_score = torch.matmul(query, key)
        # attention_score: [batch_size, num_heads, seq_len, seq_len]
        scaled_attention_score = attention_score / math.sqrt(self.params.d_k)
        # scaled_attention_score: [batch_size, num_heads, seq_len, seq_len]
        attention = self.softmax_attention(scaled_attention_score)
        # attention: [batch_size, num_heads, seq_len, seq_len]
        context_vector = torch.matmul(attention, value)
        # context_vetor: [batch_size, num_heads, seq_len, d_v]

        # 通过linear将多头信息合并
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        # context_vector: [batch_size, seq_len, num_heads * d_v]
        output = self.linear_o(context_vector)
        # output: [batch_size, seq_len, d_model]

        return output


class Feedforward_network(nn.Module):
    def __init__(self, params):
        '''
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        self.linear_FFN_1 = nn.Linear(self.params.d_model, self.params.d_ff)
        self.relu_between = nn.ReLU()
        self.linear_FFN_2 = nn.Linear(self.params.d_ff, self.params.d_model)

    def forward(self, indices):
        '''
        indices: [batch_size, seq_len, d_model]
        return indices: [batch_size, seq_len, d_model]
        '''

        # 两个FFN,维度从d_model到d_ff再回到d_model
        indices = self.linear_FFN_1(indices)
        # indices: [batch_size, seq_len, d_ff]
        indices = self.relu_between(indices)
        # indices: [batch_size, seq_len, d_ff]
        indices = self.linear_FFN_2(indices)
        # indices: [batch_size, seq_len, d_model]

        return indices
