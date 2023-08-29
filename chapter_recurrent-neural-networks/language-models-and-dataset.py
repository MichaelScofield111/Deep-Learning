import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine()) #list[[词],...[]]
corpus = [token for line in tokens
                   for token in line]
vocab = d2l.Vocab(tokens)
print(vocab.token_freqs[:10])

#stop world的图
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token:x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
plt.show()

#二元
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]

#三元
trigram_tokens = [
    triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])
]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])

plt.show()
