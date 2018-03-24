# coding: utf-8

"""自作Tokenizer
"""

from collections import Counter, defaultdict

class MyTokenizer:
    def __init__(self, num_words=10000, oov_token="<UNK>"):
        self._num_words = num_words
        self._oov_token = oov_token
        self._counter = Counter()
        self._word_index = None
        self._index_word = None

    def fit_on_texts(self, texts):
        # word count
        for text in texts:
            for word in text.split(' '):
                self._counter[word] += 1

        # top n words
        top_words = self._counter.most_common(self._num_words)
        
        word_index = {}
        index_word = {}
        for i, (word, _) in enumerate(top_words):
            word_index[word] = i + 1
            index_word[i+1] = word

        # OOV用
        word_index[self._oov_token] = i + 1
        index_word[i+1] = self._oov_token

        # 確定
        self._word_index = word_index
        self._index_word = index_word

        
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split(' '):
                try:
                    sequence.append(self._word_index[word])
                except KeyError:
                    sequence.append(self._word_index[self._oov_token])
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
