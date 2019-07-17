### 模型

import torch
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

        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # embedding层,将索引转换为词向量
        self.embedder = nn.Embedding(self.vocab_size, params.embedding_size)

        # 输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        self.output = nn.Linear(params.embedding_size, self.vocab_size)

    def forward(self, train_input_indices, train_output_indices):
        '''
        train_input_indices: [batch_size, input_seq_len]
        train_output_indices: [batch_size, output_seq_len]
        '''

        train_input_indices = self.embedder(train_input_indices)
        # train_input_indices: [batch_size, input_seq_len, embedding_size]

        train_output_indices = self.embedder(train_output_indices)
        # train_output_indices: [batch_size, output_seq_len, embedding_size]

        train_output_indices = self.output(train_output_indices)
        # train_output_indices: [batch_size, output_seq_len, vocab_size]

        train_output_indices = F.softmax(train_output_indices)
        # train_output_indices: [batch_size, output_seq_len, vocab_size]

        train_output_indices = train_output_indices.permute(0, 2, 1)
        # train_output_indices: [batch_size, vocab_size, output_seq_len]

        return train_output_indices
