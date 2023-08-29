import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'),'r') as f:
        lines = f.readlines()
    #使用 strip() 方法去除每行前后的空白字符
    #将处理后的每一行作为列表的一个元素返回。
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符标记"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知令牌类型: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


class Vocab:
    """文本词汇表 -> 字符串变成 -> 数字索引"""
    #如果一个token 少于min_freq = 0我们丢掉这个token
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    #这里的tokens是1D或2D的列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        #将词元列表展平成一个列表
        tokens = [ token for line in tokens for token in line]
    return collections.Counter(tokens)


vocab = Vocab(tokens)
#高频词给了小下标， 低频词给了大下标
print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('world:', tokens[i])
    print('indices:', vocab[tokens[i]])

def load_corups_time_machine(max_tokens=-1):
    lines = read_time_machine() #list
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token]  #vocab[token]时，Python 会调用对象的 __getitem__
              for line in tokens  #tokens [ [], [],... ] = line
                for token in line] # token []一个词 或者 句子
    #corpus就存的是一个list里面放这每个词的索引
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corups_time_machine()
print(len(corpus), len(vocab))


