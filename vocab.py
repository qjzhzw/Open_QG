### Vocab类（核心是self.vocab，这是一个集合，包含了所有的元素，这些元素的类型为Vocab_element）
### Vocab_element是Vocab类中的每个元素，包含word/index/freq/embedding


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
        self.vocab.append(Vocab_element(word, index, freq, embedding))
        self.word2index[word] = index
        self.index2word[index] = word

    # Vocab类的大小,即包含了多少[单词,索引]这样的元素
    def __len__(self):
        return len(self.vocab)

    # 判断Vocab类中是否包含某个单词
    def has_word(self, word):
        if word in self.word2index.keys():
            return True
        else:
            return False

    # 将一个句子,从单词序列转换为索引形式
    def convert_sentence2index(self, sentence):
        indices = []
        for word in sentence:
            indices.append(self.word2index[word])
        return indices

    # 将一个句子,从索引序列转换为单词形式
    def convert_index2sentence(self, indices):
        sentence = []
        for index in indices:
            sentence.append(self.index2word[index])
        return sentence


class Vocab_element():
    def __init__(self, word=None, index=None, freq=None, embedding=None):
        self.word = word
        self.index = index
        self.freq = freq
        self.embedding = embedding
