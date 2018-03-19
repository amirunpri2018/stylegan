# coding: utf-8

import logging
from logging import getLogger

import MeCab
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

AUTHORS = ['吉川英治', '宮本百合子', '豊島与志雄', '海野十三', '坂口安吾', '岡本綺堂',
           '森鴎外', '夏目漱石', '岸田国士', '中里介山', '泉鏡花', '太宰治', '国枝史郎',
           '夢野久作', '島崎藤村', '芥川龍之介', '牧野信一', '三好十郎', '林不忘', '戸坂潤',
           'wikipedia']


def load_tokenizer(lines):
    tokenizer = Tokenizer(filters="", oov_token="<unk>")
    whole_texts = []
    for line in lines:
        whole_texts.append("<s> " + line.strip() + " </s>")
    tokenizer.fit_on_texts(whole_texts)
    return tokenizer


def load_dataset(aozora, wikipedia, words_length=100, chars_length=200, sample_size=100000):
    # データセットの読み込み
    aozora_df = pd.read_csv(aozora)
    wikipedia_df = pd.read_csv(wikipedia)
    df = pd.concat([aozora_df, wikipedia_df])
    try:
        df = df.sample(sample_size * 3)
    except ValueError:
        pass

    # Author Mask
    author_mask = df.author.apply(lambda x: x in AUTHORS)
    df = df[author_mask].copy()

    # chars level
    df['chars'] = df.text.apply(lambda x: ' '.join(x))
    char_tokenizer = load_tokenizer(df.chars)

    # pos level
    mecab = MeCab.Tagger('-O chasen')
    df['chasen'] = df.text.apply(lambda x: mecab.parse(x).strip())
    df['pos'] = df.chasen.apply(lambda x: ' '.join(
        [word.split('\t')[3] for word in x.split('\n')[:-1]]))
    pos_tokenizer = load_tokenizer(df.pos)

    # 非効率かもしれないが、こちらでlengthによる絞り込みを行う。
    char_len_mask = df.chars.apply(lambda x: x.count(' ') <= chars_length-3)
    words_len_mask = df.pos.apply(lambda x: x.count(' ') <= words_length-3)  # <s>, </s>のぶん
    df = df[words_len_mask & char_len_mask].copy()
    
    try:
        df = df.sample(sample_size)
    except ValueError:
        pass

    train_C, test_C, train_P, test_P, train_A, test_A = train_test_split(
        df.chars, df.pos, df.author, test_size=0.1, random_state=42
    )

    return (train_C, test_C), (train_P, test_P), (train_A, test_A), (char_tokenizer, pos_tokenizer)


def batch_generator(T, A, tokenizer, batch_size=200, max_len=100, shuffle_flag=True):
    """generatorのpretraing用データローダー

    Arguments:
        T -- テキスト
        A -- 著者
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
                line = ' '.join(line)
                whole_texts.append("<s> " + line.strip() + " </s>")

            x = tokenizer.texts_to_sequences(whole_texts)
            x = pad_sequences(x, padding='post', maxlen=max_len)

            yield x, y


def pretrain_batch_generator(C, A, char_tokenizer, batch_size=200, max_len=200):
    for chars, author in batch_generator(C, A, char_tokenizer, batch_size, max_len):
        train_target = np.hstack(
            (chars[:, 1:], np.zeros((len(chars), 1), dtype=np.int32)))
        yield [chars, author, chars], np.expand_dims(train_target, -1)
