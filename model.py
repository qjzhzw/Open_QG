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

        input_indices = self.encoder(input_indices)
        output_indices = self.decoder(output_indices, input_indices)

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

        # 构造掩膜和位置信息
        self.mask_and_position = Mask_and_position(self.params)

        # embedding层,将索引/位置信息转换为词向量
        self.word_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.position_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)

        # 多个相同子结构组成的encoder子层,层数为num_layers
        self.encoder_layers = nn.ModuleList([Encoder_layer(self.params) for _ in range(self.params.num_layers)])

        # 测试所使用GRU
        self.GRU_encoder = nn.GRU(self.params.d_model, self.params.d_model, batch_first=True, num_layers=1)

    def forward(self, input_indices):
        '''
        input_indices: [batch_size, input_seq_len]
        return input_indices: [batch_size, input_seq_len]
        '''

        # 构造掩膜和位置信息
        input_indices_masks = self.mask_and_position.build_masks(input_indices).unsqueeze(2)
        input_indices_positions = self.mask_and_position.build_positions(input_indices)
        # input_indices_masks: [batch_size, input_seq_len, 1]
        # input_indices_positions: [batch_size, input_seq_len]

        # 将索引/位置信息转换为词向量
        input_indices = self.word_embedding_encoder(input_indices) + self.position_embedding_encoder(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        # 经过多个相同子结构组成的decoder子层,层数为num_layers
        for encoder_layer in self.encoder_layers:
            input_indices = encoder_layer(input_indices, input_indices_masks)
        # input_indices: [batch_size, input_seq_len, d_model]

        # # 测试所使用GRU
        # input_indices = self.word_embedding_encoder(input_indices)
        # # input_indices: [batch_size, input_seq_len, d_model]
        # input_indices, _ = self.GRU_encoder(input_indices)
        # # input_indices: [batch_size, input_seq_len, d_model]

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

        # 构造掩膜和位置信息
        self.mask_and_position = Mask_and_position(self.params)

        # embedding层,将索引/位置信息转换为词向量
        self.word_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.position_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)

        # 多个相同子结构组成的decoder子层,层数为num_layers
        self.decoder_layers = nn.ModuleList([Decoder_layer(self.params) for _ in range(self.params.num_layers)])

        # 输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        self.output = nn.Linear(self.params.d_model, self.vocab_size)

        # 测试所使用GRU
        self.GRU_decoder = nn.GRU(self.params.d_model, self.params.d_model, batch_first=True, num_layers=1)

    def forward(self, output_indices, input_indices):
        '''
        output_indices: [batch_size, output_seq_len]
        input_indices: [batch_size, input_seq_len, d_model]
        return output_indices: [batch_size, vocab_size, output_seq_len]
        '''

        # 构造掩膜和位置信息
        output_indices_masks = self.mask_and_position.build_masks(output_indices).unsqueeze(2)
        output_indices_positions = self.mask_and_position.build_positions(output_indices)
        # output_indices_masks: [batch_size, output_seq_len, 1]
        # output_indices_positions: [batch_size, output_seq_len]

        # 将索引/位置信息转换为词向量
        output_indices = self.word_embedding_decoder(output_indices) + self.position_embedding_decoder(output_indices)
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过多个相同子结构组成的decoder子层,层数为num_layers
        for decoder_layer in self.decoder_layers:
            output_indices = decoder_layer(output_indices, input_indices, output_indices_masks)
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        output_indices = self.output(output_indices)
        # output_indices: [batch_size, output_seq_len, vocab_size]
        output_indices = output_indices.permute(0, 2, 1)
        # output_indices: [batch_size, vocab_size, output_seq_len]

        # # 测试所使用GRU
        # output_indices = self.word_embedding_decoder(output_indices)
        # # output_indices: [batch_size, output_seq_len, d_model]
        # input_indices = input_indices[:, 0, :].unsqueeze(0).contiguous()
        # # input_indices: [1, batch_size, d_model]
        # output_indices, _ = self.GRU_decoder(output_indices, input_indices)
        # # output_indices: [batch_size, output_seq_len, d_model]
        # output_indices = self.output(output_indices)
        # # output_indices: [batch_size, output_seq_len, vocab_size]
        # output_indices = output_indices.permute(0, 2, 1)
        # # output_indices: [batch_size, vocab_size, output_seq_len]

        return output_indices


class Encoder_layer(nn.Module):
    def __init__(self, params):
        '''
        params: 参数集合
        '''
        super().__init__()

        self.params = params

        # self_attention结构
        self.self_attention = Multihead_attention(self.params)

        # FFN结构
        self.feedforward_network = Feedforward_network(self.params)

    def forward(self, input_indices, input_indices_masks):
        '''
        input_indices: [batch_size, input_seq_len, d_model]
        input_indices_masks: [batch_size, input_seq_len, 1]
        return input_indices: [batch_size, input_seq_len, d_model]
        '''

        # 经过self_attention结构
        input_indices = self.self_attention(query=input_indices,
                                            key=input_indices,
                                            value=input_indices)
        input_indices *= input_indices_masks
        # input_indices: [batch_size, input_seq_len, d_model]

        # 经过FFN结构
        input_indices = self.feedforward_network(input_indices)
        input_indices *= input_indices_masks
        # input_indices: [batch_size, input_seq_len, d_model]

        return input_indices


class Decoder_layer(nn.Module):
    def __init__(self, params):
        '''
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

    def forward(self, output_indices, input_indices, output_indices_masks):
        '''
        output_indices: [batch_size, output_seq_len, d_model]
        input_indices: [batch_size, input_seq_len, d_model]
        output_indices_masks: [batch_size, output_seq_len, 1]
        return output_indices: [batch_size, output_seq_len, d_model]
        '''

        # 经过self_attention结构
        output_indices = self.self_attention(query=output_indices,
                                             key=output_indices,
                                             value=output_indices)
        output_indices *= output_indices_masks
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过mutual_attention结构
        output_indices = self.mutual_attention(query=output_indices,
                                               key=input_indices,
                                               value=input_indices)
        output_indices *= output_indices_masks
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过FFN结构
        output_indices = self.feedforward_network(output_indices)
        output_indices *= output_indices_masks
        # output_indices: [batch_size, output_seq_len, d_model]

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

        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.d_model)

    def forward(self, query, key, value):
        '''
        query: [batch_size, seq_len, d_model]
        key: [batch_size, seq_len, d_model]
        value: [batch_size, seq_len, d_model]
        return indices: [batch_size, seq_len, d_model]
        '''
        # 保留进入子层时的输入状态,用于离开子层时的残差结构
        residual = query

        # 断言: query/key/value的batch_size和seq_len均一致
        assert query.size(0) == key.size(0) == value.size(0)
        assert key.size(1) == value.size(1)

        batch_size = query.size(0)
        seq_len_query = query.size(1)
        seq_len_key = key.size(1)

        # 将输入通过三个各自的linear转换为query/key/value
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

        # attention的计算,通过attention分数得到context_vector向量
        attention_score = torch.matmul(query, key)
        # attention_score: [batch_size, num_heads, seq_len_query, seq_len_key]
        scaled_attention_score = attention_score / math.sqrt(self.params.d_k)
        # scaled_attention_score: [batch_size, num_heads, seq_len_query, seq_len_key]
        attention = self.softmax_attention(scaled_attention_score)
        # attention: [batch_size, num_heads, seq_len_query, seq_len_key]
        context_vector = torch.matmul(attention, value)
        # context_vetor: [batch_size, num_heads, seq_len_query, d_v]

        # 通过linear将多头信息合并
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_query, -1)
        # context_vector: [batch_size, seq_len_query, num_heads * d_v]
        indices = self.linear_o(context_vector)
        # indices: [batch_size, seq_len_query, d_model]

        # dropout
        indices = self.dropout(indices)
        # 残差结构
        indices += residual
        # 层正则化
        indices = self.layer_norm(indices)

        return indices


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

        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.d_model)

    def forward(self, indices):
        '''
        indices: [batch_size, seq_len, d_model]
        return indices: [batch_size, seq_len, d_model]
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

        return indices


class Mask_and_position():
    def __init__(self, params):
        self.params = params

    def build_masks(self, indices):
        '''
        indices: [batch_size, seq_len]
        '''

        # 构造掩膜,是序列的部分填1,非序列的部分填0
        indices_masks = []
        for indice in indices:
            indice_mask = []
            for index in indice:
                if index == 0:
                    indice_mask.append(0)
                else:
                    indice_mask.append(1)
            indices_masks.append(indice_mask)

        # 将二维list的batch直接转换为tensor的形式
        indices_masks = torch.tensor(indices_masks).float()

        # 如果参数中设置了使用cuda且当前cuda可用,则将数据放到cuda上
        if self.params.cuda:
            indices_masks = indices_masks.cuda()

        return indices_masks

    def build_positions(self, indices):
        '''
        indices: [batch_size, seq_len]
        '''

        # 构造位置信息,位置从1开始
        indices_positions = []
        for indice in indices:
            indice_position = []
            for position, index in enumerate(indice):
                indice_position.append(position)
            indices_positions.append(indice_position)

        # 将二维list的batch直接转换为tensor的形式
        indices_positions = torch.tensor(indices_positions).float()

        # 如果参数中设置了使用cuda且当前cuda可用,则将数据放到cuda上
        if self.params.cuda:
            indices_positions = indices_positions.cuda()

        return indices_positions
