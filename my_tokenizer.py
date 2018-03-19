# coding: utf-8

"""自作Tokenizer
"""

from collections import Counter, defaultdict

class MyTokenizer:
    def __init__(self, num_words=10000, oov_token="<UNK>"):
        self._num_words = num_words
        self._oov_token = oov_token
        self._counter = Counter()
        self._word_index = defaultdict(lambda: -1)
        self._index_word = {}

    def fit_on_texts(self, texts):
        # word count
        for text in texts:
            for word in text.split(' '):
                self._counter[word] += 1

        # top n words
        top_words = self._counter.most_common(self._num_words)
        
        for i, (word, _) in enumerate(top_words):
            self._word_index[word] = i + 1
            self._index_word[i+1] = word

        # OOV用
        self._word_index[self._oov_token] = -1
        self._index_word[-1] = self._oov_token

        
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split(' '):
                sequence.append(self._word_index[word])
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences, return_words=False):
        texts = []
        for sequence in sequences:
            words = [self._index_word[i] for i in sequence if i > 0]
            texts.append(words)
        if return_words:
            return texts
        else:
            return [' '.join(words) for words in texts]

    @property
    def word_index(self):
        return self._word_index
