# coding: utf-8

from collections import Counter

import MeCab
import numpy as np

mecab = MeCab.Tagger("-O chasen")


class MonteCarloSearchNode:

    def __init__(self, parent_node, cond_tokens, token,
                 encoded_seq, y, word_decoder, attention_model, discriminator,
                 word_tokenizer, char_tokenizer, pos_tokenizer, states_value, sample_size=10, sampled_n=1):

        self.parent = parent_node  # 親のノード rootノードはNone

        self.sampled_n = sampled_n  # 親ノードからサンプリングされた回数
        self.cond_tokens = cond_tokens  # いままでのtokenのリスト
        self.children = []  # 子ノード
        self.token = token  # このノードのtoken
        self.states_value = states_value

        # 木の中で共通
        self.word_decoder = word_decoder
        self.attention_model = attention_model
        self.discriminator = discriminator
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        self.pos_tokenizer = pos_tokenizer
        self.sample_size = sample_size
        self.encoded_seq = encoded_seq
        self.y = y  # 作家情報

        # 計算済みのQ値
        self.qvalue_ = None

        # 終端ノードであるかを計算
        # words_len
        if self.token == self.word_tokenizer.word_index['</s>'] or len(self.cond_tokens) >= 100 - 1:
            self.is_end = True
        else:
            self.is_end = False

    @property
    def _is_root(self):
        return self.parent is None

    def search(self):
        """次のtokenをサンプリングして子ノードを作るを再帰的に繰り返す。
        """
        if self.is_end:
            return

        # Decode
        decoded_seq, *states_value = self.word_decoder.predict(
            [np.array([self.token])] + self.states_value
        )
        # Attention
        output_tokens, _ = self.attention_model.predict(
            [self.encoded_seq, decoded_seq, np.array([self.y])]  # 条件がここにつく
        )
        # sampling
        sampled_tokens = np.random.choice(
            range(output_tokens.shape[2]), p=output_tokens[0, -1], size=self.sample_size)

        # create children
        for sampled_token, sampled_n in Counter(sampled_tokens).items():
            self.children.append(MonteCarloSearchNode(
                parent_node=self, cond_tokens=self.cond_tokens+[sampled_token], token=sampled_token,
                encoded_seq=self.encoded_seq, y=self.y, word_decoder=self.word_decoder,
                attention_model=self.attention_model, discriminator=self.discriminator, word_tokenizer=self.word_tokenizer,
                char_tokenizer=self.char_tokenizer, pos_tokenizer=self.pos_tokenizer, states_value=states_value, sampled_n=sampled_n)
            )  # 子ノードの追加

        # search for children
        for node in self.children:
            node.serach()

    def qvalue(self):
        """Q値
        終端ノードであれば、Discriminatorの評価値をそのまま使用する。
        そうでなければ、子ノードのQ値の重み付き平均を求めて平均する
        """
        if self.qvalue_ is not None:
            # 計算済みのQ値を返す
            return self.qvalue_
        else:
            if self.is_end:
                words = self.word_tokenizer.sequences_to_texts([self.cond_tokens])[
                    0]
                text = ''.join(words)

                char_input = ' '.join(text)
                chasen = mecab.parse(text).strip()
                pos_input = [word.split('\t')[3]
                             for word in chasen.split('\n')[:-1]]
                condition_input = [self.y]

                self.qvalue_ = self.discriminator(
                    [char_input, pos_input, condition_input])  # discriminatorによる評価
            else:
                self.qvalue_ = np.sum(
                    [node.sampled_n * node.qvalue() for node in self.children]) / self.sample_size
            return self.qvalue_
