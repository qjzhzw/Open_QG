### Vocab类（核心是self.vocab，这是一个集合，包含了所有的元素，这些元素的类型为Vocab_element）
### Vocab_element是Vocab类中的每个元素，包含word/index/freq/embedding

import torch

class Vocab():
    def __init__(self):
        self.vocab = []
        self.word2index = {}
        self.index2word = {}

        # 将常数加入到vocab中
        self.constants = ['<pad>', '<unk>', '<s>', '</s>', '<cls>', '<sep>']
        for index, constant in enumerate(self.constants):
            self.add_element(constant, index)

    # 在Vocab类中加入一个元素
    # 必选项:单词,索引
    # 可选性:词频,词向量
    def add_element(self, word, index, freq=None, embedding=None):
        '''
        word: 需要添加元素的单词
        index: 需要添加元素的索引
        freq: 需要添加元素的词频(可选)
        embedding: 需要添加元素的词向量(可选)
        '''

        self.vocab.append(Vocab_element(word, index, freq, embedding))
        self.word2index[word] = index
        self.index2word[index] = word

    # Vocab类的大小,即包含了多少[单词,索引]这样的元素
    def __len__(self):
        return len(self.vocab)

    # 判断Vocab类中是否包含某个单词
    def has_word(self, word):
        '''
        word: 输入单词
        '''

        if word in self.word2index.keys():
            return True
        else:
            return False

    # 将一个单词转换为索引
    def convert_word2index(self, word):
        '''
        word: 输入单词
        return index: 输出索引
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
        index: 输入索引
        return index: 输出单词
        '''

        # 如果索引是合法的(即小于vocab的大小),则返回该索引的单词,否则返回<unk>
        word = None
        if index >= 0 and index < len(self):
            word = self.index2word[index]
        else:
            word = '<unk>'
        return word

    # 将一个句子,从单词序列转换为索引形式
    def convert_sentence2index(self, sentence):
        '''
        sentence: 输入单词序列
        return indices: 输出索引序列
        '''

        # 如果输入是tensor形式,转换为numpy形式
        if torch.is_tensor(sentence):
            sentence = sentence.cpu().numpy()

        # 通过遍历的方式,将单词序列转换为索引形式
        indices = []
        for word in sentence:
            indices.append(self.convert_word2index(word))
        return indices

    # 将一个句子,从索引序列转换为单词形式
    def convert_index2sentence(self, indices, mode=False):
        '''
        indices: 输入索引序列
        mode: True表示输出完整序列
              False表示遇到</s>就停止(只输出到</s>前的序列)
        return sentence: 输出单词序列
        '''

        # 如果输入是tensor形式,转换为numpy形式
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()

        # 通过遍历的方式,将单词序列转换为索引形式
        sentence = []
        for index in indices:
            # 在mode为False时,遇到</s>就停止
            if index == self.convert_word2index('</s>') and mode == False:
                break
            else:
                sentence.append(self.convert_index2word(index))
        return sentence


class Vocab_element():
    def __init__(self, word=None, index=None, freq=None, embedding=None):
        self.word = word
        self.index = index
        self.freq = freq
        self.embedding = embedding
