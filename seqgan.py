# coding: utf-8

import numpy as np
import pandas as pd

from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

from dataloader import load_dataset, batch_generator, generator_pretrain_batch, discriminator_pretrain_batch
from model import get_discriminator, get_generator


WORDS_LEN = 50
CHARS_LEN = 100


def generate_sequence(encoder, word_decoder, attention_model, word_tokenizer,
                      texts=["私 は 猫 です 。", "僕 は 犬 です 。"], authors=["夏目漱石", "森鴎外"]):

    bos_eos = word_tokenizer.texts_to_sequences(["<s>", "</s>"])

    for X, Y in batch_generator(pd.Series(texts), pd.Series(authors),
                                word_tokenizer, batch_size=len(texts), max_len=WORDS_LEN):
        break

    sequences = []
    for text, author, x, y in zip(texts, authors, X, Y):
        x = np.reshape(x, (1, -1))
        y = np.reshape(y, (1, -1))
        target_seq = np.array(bos_eos[0])
        output_seq = bos_eos[0][:]
        attention_seq = np.empty((0, WORDS_LEN))
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

            if (sampled_token_index == bos_eos[1] or len(output_seq) > WORDS_LEN):
                break

            target_seq = np.array(sampled_token_index)
            prev_token_index = sampled_token_index

        sequences.append(output_seq)

    words_list = word_tokenizer.sequences_to_texts(
        sequences, return_words=True)

    results = []
    for text, author, words in zip(texts, authors, words_list):
        results.append((text, author, ''.join(words)))

    return results


class PretrainGeneratorCallBack(Callback):

    def __init__(self, encoder, word_decoder, attention, word_tokenizer):
        self._encoder = encoder
        self._woed_decoder = word_decoder
        self._attention = attention
        self._tokenizer = word_tokenizer

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        results = generate_sequence(
            self._encoder, self._woed_decoder, self._attention, self._tokenizer)
        for result in results:
            print(result)


def pretrain_generator(generator, train_W, test_W, train_A, test_A, word_tokenizer, encoder, word_decoder, attention):
    """Generatorの事前学習
    """
    generator.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy')

    checkpoint_cb = ModelCheckpoint(
        "models/generator-weights.{epoch:02d}.hdf5", period=10)
    simple_test_cb = PretrainGeneratorCallBack(
        encoder, word_decoder, attention, word_tokenizer)

    generator.fit_generator(
        generator=generator_pretrain_batch(
            train_W, train_A, word_tokenizer, batch_size=200, words_len=WORDS_LEN),
        steps_per_epoch=100,
        epochs=50, verbose=2,
        validation_data=generator_pretrain_batch(
            test_W, test_A, word_tokenizer, batch_size=200, words_len=WORDS_LEN),
        validation_steps=1,
        callbacks=[checkpoint_cb, simple_test_cb]
    )

def train_generator(generator, train_W, test_W, train_A, test_A, word_tokenizer, encoder, word_decoder, attention):
    """Generatorの事前学習
    """
    generator.fit_generator(
        generator=generator_pretrain_batch(
            train_W, train_A, word_tokenizer, batch_size=200, words_len=WORDS_LEN),
        steps_per_epoch=100,
        epochs=5, verbose=2,
        validation_data=generator_pretrain_batch(
            test_W, test_A, word_tokenizer, batch_size=200, words_len=WORDS_LEN),
        validation_steps=1
    )

def pretrain_discriminator(discriminator, train_W, test_W, train_A, test_A,
                           word_tokenizer, char_tokenizer, pos_tokenizer,
                           encoder, word_decoder, attention_model):
    """Discriminatorの事前学習
    """
    discriminator.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_cb = ModelCheckpoint(
        "models/discriminator-weights.{epoch:02d}.hdf5", period=1)

    discriminator.fit_generator(
        generator=discriminator_pretrain_batch(
            train_W, train_A, word_tokenizer, char_tokenizer, pos_tokenizer,
            encoder, word_decoder, attention_model,
            chars_len=CHARS_LEN, pos_len=WORDS_LEN, batch_size=200, shuffle_flag=True),
        steps_per_epoch=10,
        epochs=10,
        verbose=2,
        validation_data=discriminator_pretrain_batch(
            test_W, test_A, word_tokenizer, char_tokenizer, pos_tokenizer,
            encoder, word_decoder, attention_model,
            chars_len=CHARS_LEN, pos_len=WORDS_LEN, batch_size=200, shuffle_flag=True),
        validation_steps=1,
        callbacks=[checkpoint_cb]
    )

def train_discriminator(discriminator, train_W, test_W, train_A, test_A,
                           word_tokenizer, char_tokenizer, pos_tokenizer,
                           encoder, word_decoder, attention_model):
    """Discriminatorの事前学習
    """
    discriminator.fit_generator(
        generator=discriminator_pretrain_batch(
            train_W, train_A, word_tokenizer, char_tokenizer, pos_tokenizer,
            encoder, word_decoder, attention_model,
            chars_len=CHARS_LEN, pos_len=WORDS_LEN, batch_size=200, shuffle_flag=True),
        steps_per_epoch=10,
        epochs=1,
        verbose=2,
        validation_data=discriminator_pretrain_batch(
            test_W, test_A, word_tokenizer, char_tokenizer, pos_tokenizer,
            encoder, word_decoder, attention_model,
            chars_len=CHARS_LEN, pos_len=WORDS_LEN, batch_size=200, shuffle_flag=True),
        validation_steps=1
    )
