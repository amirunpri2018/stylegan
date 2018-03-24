# coding: utf-8

import logging
from logging import getLogger

import MeCab
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
from my_tokenizer import MyTokenizer as Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from mcsearch import MonteCarloSearchNode

import random

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

AUTHORS = ['吉川英治', '宮本百合子', '豊島与志雄', '海野十三', '坂口安吾', '岡本綺堂',
           '森鴎外', '夏目漱石', '岸田国士', '中里介山', '泉鏡花', '太宰治', '国枝史郎',
           '夢野久作', '島崎藤村', '芥川龍之介', '牧野信一', '三好十郎', '林不忘', '戸坂潤',
           'wikipedia']

mecab = MeCab.Tagger('-O chasen')


def random_unkowned_text(wakati, rate=0.05):
    words = wakati.split(' ')
    mask = np.random.choice([True, False], size=len(words), p=[0.05, 0.95])
    text = ''.join([w if m else '<UNK>' for (w, m) in zip(words, mask)])
    return text


def load_tokenizer(lines):
    """Tokenizerの作成

    Arguments:
        lines {pandas.Series} -- Series of Text

    Returns:
        Tokenizer -- fit済みtokenizer
    """

    tokenizer = Tokenizer(num_words=10000, oov_token="<UNK>")
    whole_texts = []
    for line in lines:
        whole_texts.append("<s> " + line.strip() + " </s>")
    tokenizer.fit_on_texts(whole_texts)
    return tokenizer


def load_dataset(datasets, words_len=100, chars_len=100, sample_size=500000):
    """青空文庫とwikipediaのデータからデータセットを作成する

    Arguments:
        datasets {list of str} -- データセットへのパス

    Keyword Arguments:
        words_len {int} -- [description] (default: {50})
        chars_len {int} -- [description] (default: {100})
        sample_size {int} -- [description] (default: {100000})

    Returns:
        CharsDataset, WordsDataset, POSDataset, tokenizers -- 構築したもののタプル
    """

    # データセットの読み込み
    dfs = []
    for dataset in datasets:
        dfs.append(pd.read_csv(dataset))
    df = pd.concat(dfs)

    # Author Mask
    author_mask = df.author.apply(lambda x: x in AUTHORS)
    df = df[author_mask]

    # Char len Mask
    char_mask = df.text.apply(lambda x: len(x) < chars_len)
    df = df[char_mask]

    # chars level
    df['chars'] = df.text.apply(lambda x: ' '.join(x))
    char_tokenizer = load_tokenizer(df.chars)

    # words level and pos level
    mecab = MeCab.Tagger('-O chasen')
    df['chasen'] = df.text.apply(lambda x: mecab.parse(x).strip())
    df['wakati'] = df.chasen.apply(lambda x: ' '.join(
        [word.split('\t')[0] for word in x.split('\n')[:-1]]))
    df['pos'] = df.chasen.apply(lambda x: ' '.join(
        [word.split('\t')[3] for word in x.split('\n')[:-1]]))
    word_tokenizer = load_tokenizer(df.wakati)
    pos_tokenizer = load_tokenizer(df.pos)

    # words len mask
    words_len_mask = df.wakati.apply(
        lambda x: x.count(' ') < words_len-1)  # <s>, </s>のぶん
    char_len_mask = df.chars.apply(lambda x: x.count(' ') <= chars_len-1)
    df = df[words_len_mask & char_len_mask].copy()

    try:
        df = df.sample(sample_size)
    except ValueError:
        pass

    train_C, test_C, train_W, test_W, train_P, test_P, train_A, test_A = train_test_split(
        df.chars, df.wakati, df.pos, df.author, test_size=0.1
    )

    return (train_C, test_C), (train_W, test_W), (train_P, test_P), (train_A, test_A), (char_tokenizer, word_tokenizer, pos_tokenizer)


def batch_generator(T, A, tokenizer, batch_size=200, max_len=100, shuffle_flag=True):
    """バッチの生成

    Arguments:
        T {pd.Series} -- テキスト（分かち書き済み）
        A {pd.Series} -- 著者
        tokenizer {Tokenizer} -- tokenizer

    Keyword Arguments:
        batch_size {int} -- バッチサイズ (default: {200})
        max_len {int} -- 最長のトークン列長 (default: {100})
        shuffle_flag {bool} -- 入力されたデータをシャッフルするかいなかのフラグ (default: {True})
    """

    data_size = T.shape[0]
    A = A.apply(lambda x: AUTHORS.index(x))
    A = np_utils.to_categorical(A, num_classes=len(AUTHORS))

    while True:
        if shuffle_flag:
            T, A = shuffle(T, A)

        for i in range(data_size//batch_size):
            x = T[i*batch_size: (i+1)*batch_size]
            y = A[i*batch_size: (i+1)*batch_size]

            whole_texts = []
            for line in x:
                whole_texts.append("<s> " + line.strip() + " </s>")
            x = tokenizer.texts_to_sequences(whole_texts)
            x = pad_sequences(x, padding='post', maxlen=max_len)

            yield x, y


def generator_pretrain_batch(W, A, word_tokenizer, batch_size=200, words_len=100):
    """Generatorの事前学習のためのバッチ生成

    Arguments:
        W {pd.Series} -- 分かち書き済みの単語系列
        A {pd.Series} -- 著者
        word_tokenizer {Tokenizer} -- 単語レベルのtokenizer

    Keyword Arguments:
        batch_size {int} -- バッチサイズ (default: {200})
        words_len {int} -- 最長単語長 (default: {100})
    """

    for words, author in batch_generator(W, A, word_tokenizer, batch_size, words_len):
        author = np.reshape(np.repeat(author, words_len, axis=0),
                            (author.shape[0], words_len, author.shape[1]))
        train_target = np.hstack(
            (words[:, 1:], np.zeros((len(words), 1), dtype=np.int32)))
        yield [words, words, author], np.expand_dims(train_target, -1)


def text_to_pos(text):
    chasen = mecab.parse(text).strip()
    pos = ' '.join([word.split('\t')[3] for word in chasen.split('\n')[:-1]])
    return pos


def discriminator_pretrain_batch(W, A, word_tokenizer, char_tokenizer, pos_tokenizer,
                                 encoder, word_decoder, attention_model,
                                 chars_len=100, pos_len=50, batch_size=200, shuffle_flag=True):
    """Discriminatorの事前学習のためのバッチ生成
    """

    data_size = W.shape[0]

    while True:
        if shuffle_flag:
            W, A = shuffle(W, A)

        for i in range(data_size//batch_size):

            true_fake_mask = (np.random.random(size=batch_size) > 0.5)

            w = W[i*batch_size: (i+1)*batch_size]  # wakati
            a = A[i*batch_size: (i+1)*batch_size]

            batch_text = []
            batch_y = []

            # generate nagative sample
            fake_authors = np.random.choice(
                AUTHORS, size=batch_size - true_fake_mask.sum()).tolist()
            fake_wakati_list = generate_fake_text(
                encoder, word_decoder, attention_model, word_tokenizer,
                w[~true_fake_mask],
                pd.Series(fake_authors)
            )

            for fake_wakati in fake_wakati_list:
                fake_text = ''.join(fake_wakati)
                batch_text.append(fake_text)
                batch_y.append(0)

            # calc unk rate
            words_num = 0
            unk_num = 0
            for line in fake_wakati_list:
                words_num += len(line)
                unk_num += line.count('<UNK>')
            unk_rate = unk_num / words_num

            batch_author = pd.Series(fake_authors + a[true_fake_mask].tolist())
            for wakati in w[true_fake_mask]:
                true_text = random_unkowned_text(wakati, unk_rate)
                batch_text.append(true_text)
                batch_y.append(1)

            whole_chars = []
            whole_poss = []
            for text in batch_text:
                chars = ' '.join(text)
                whole_chars.append("<s> " + chars.strip() + " </s>")
                pos = text_to_pos(text)
                whole_poss.append("<s> " + pos.strip() + " </s>")

            c = char_tokenizer.texts_to_sequences(whole_chars)
            p = pos_tokenizer.texts_to_sequences(whole_poss)
            c = pad_sequences(c, padding='post', maxlen=chars_len)
            p = pad_sequences(p, padding='post', maxlen=pos_len)

            a = batch_author.apply(lambda x: AUTHORS.index(x))
            a = np_utils.to_categorical(a, num_classes=len(AUTHORS))

            y = np.array(batch_y).astype(float)

            yield [c, p, a], y


def generate_fake_text(encoder, word_decoder, attention_model, word_tokenizer, W, A):

    bos_eos = word_tokenizer.texts_to_sequences(["<s>", "</s>"])

    A = A.apply(lambda x: AUTHORS.index(x))
    A = np_utils.to_categorical(A, num_classes=len(AUTHORS))

    whole_texts = []
    for wakati in W:
        whole_texts.append("<s> " + wakati.strip() + " </s>")
    X = word_tokenizer.texts_to_sequences(whole_texts)
    X = pad_sequences(X, padding='post', maxlen=50)

    sequences = []

    for x, y in zip(X, A):
        x = np.reshape(x, (1, -1))
        y = np.reshape(y, (1, -1))

        target_seq = np.array(bos_eos[0])
        output_seq = bos_eos[0][:]
        attention_seq = np.empty((0, 50))
        prev_token_index = bos_eos[0][:]  # 1つ前のトークン

        encoded_seq, *states_value = encoder.predict(x)

        while True:
            decoded_seq, * \
                states_value = word_decoder.predict(
                    [target_seq] + states_value)
            output_tokens, attention = attention_model.predict(
                [encoded_seq, decoded_seq, np.array([y])])  # condition
            sampled_token_index = [np.argmax(output_tokens[0, -1, :])]

            if prev_token_index == sampled_token_index:
                break

            output_seq += sampled_token_index
            attention_seq = np.append(attention_seq, attention[0], axis=0)

            if (sampled_token_index == bos_eos[1] or len(output_seq) > 50):
                break

            target_seq = np.array(sampled_token_index)

        sequences.append(output_seq)

    wakati_list = word_tokenizer.sequences_to_texts(
        sequences, return_words=True)

    results = []
    for wakati in wakati_list:
        results.append(wakati[1:-1])

    return results


def generate_reward_data(node):
    if node.token is None:
        return
    for node in node.children:
        x, y = node.get_reward()
        yield x, y
        for x, y in generate_reward_data(node):
            yield x, y


def generate_generator_training_data(W, A, word_tokenizer, char_tokenizer, pos_tokenizer,
                                     encoder, word_decoder, attention_model, discriminator):
    while True:
        W, A = shuffle(W, A)
        for w, a in zip(W, A):
            y = np_utils.to_categorical(
                [AUTHORS.index(a)], num_classes=len(AUTHORS))

            wakatis = ["<s> " + ' '.join(w) + " </s>"]

            inputs = word_tokenizer.texts_to_sequences(wakatis)
            inputs = pad_sequences(inputs, maxlen=50)

            # 入力文字列から初期状態を求める
            encoded_seq, *states_value = encoder.predict(inputs)

            start_depth = np.random.randint(0, min(len(wakatis) - 3, 50 - 3))
            target_seq = np.array([word_tokenizer.word_index['<s>']])
            for _ in range(start_depth):
                decoded_seq, *states_value = word_decoder.predict(
                    [target_seq] + states_value
                )
                output_tokens, _ = attention_model.predict(
                    [encoded_seq, decoded_seq, np.array([y])])  # condition
                sampled_token_index = [np.argmax(output_tokens[0, -1, :])]

                if sampled_token_index == word_tokenizer.word_index['</s>']:
                    break
                target_seq = np.array(sampled_token_index)


            mctree = MonteCarloSearchNode(
                None, [word_tokenizer.word_index['<s>']],
                word_tokenizer.word_index['<s>'],
                encoded_seq, y, word_decoder, attention_model, discriminator,
                word_tokenizer, char_tokenizer, pos_tokenizer,
                states_value, sample_size=10, sampled_n=1,
                remain_depth=3
            )

            mctree.search()

            for x, y in generate_reward_data(mctree)[:100]:  # 1文あたり100個まで
                yield x, y
