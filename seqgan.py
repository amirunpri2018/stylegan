# coding: utf-8

import numpy as np
import pandas as pd

from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

from dataloader import load_dataset, batch_generator, generator_pretrain_batch, discriminator_pretrain_batch
from model import get_discriminator, get_generator


WORDS_LEN = 100
CHARS_LEN = 200


def generate_sequence(encoder, word_decoder, attention_model, word_tokenizer,
                      texts=["私は猫です。", "僕は犬です。"], authors=["夏目漱石", "森鴎外"]):

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
            train_W, train_A, word_tokenizer, batch_size=200, words_len=200),
        steps_per_epoch=100,
        epochs=100, verbose=2,
        validation_data=generator_pretrain_batch(
            test_W, test_A, word_tokenizer, batch_size=200, words_len=200),
        validation_steps=1,
        callbacks=[checkpoint_cb, simple_test_cb]
    )


def pretrain_discriminator(discriminator, train_C, test_C, train_P, test_P,
                           train_A, test_A, char_tokenizer, pos_tokenizer):
    """Discriminatorの事前学習

    Arguments:
        discriminator {[type]} -- [description]
        train_C {[type]} -- [description]
        test_C {[type]} -- [description]
        train_P {[type]} -- [description]
        test_P {[type]} -- [description]
        train_A {[type]} -- [description]
        test_A {[type]} -- [description]
        char_tokenizer {[type]} -- [description]
        pos_tokenizer {[type]} -- [description]
    """
    discriminator.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_cb = ModelCheckpoint(
        "models/discriminator-weights.{epoch:02d}.hdf5", period=10)

    discriminator.fit_generator(
        generator=discriminator_pretrain_batch(
            train_C, train_P, train_A, char_tokenizer, pos_tokenizer, batch_size=200,
            chars_len=CHARS_LEN, pos_len=WORDS_LEN),
        steps_per_epoch=100,
        epochs=100, verbose=2,
        validation_data=discriminator_pretrain_batch(
            test_C, test_P, test_A, char_tokenizer, pos_tokenizer, batch_size=200,
            chars_len=CHARS_LEN, pos_len=WORDS_LEN),
        validation_steps=1,
        callbacks=[checkpoint_cb]
    )


def main():
    aozora = "./data/sample_aozora.csv"
    wikipedia = "./data/sample_wikipedia.csv"

    Cs, Ws, Ps, As, tokenizers = load_dataset(
        [aozora, wikipedia], WORDS_LEN, CHARS_LEN, sample_size=100000)
    char_tokenizer, word_tokenizer, pos_tokenizer = tokenizers

    # initialize
    encoder, generator, word_decoder, attention = get_generator(
        vocab_size=len(word_tokenizer.word_index)+1, emb_dim=512, hid_dim=1024, att_dim=1024,
        condition_num=21, words_len=WORDS_LEN
    )

    discriminator = get_discriminator(
        char_emb_dim=128, num_filters=64, filter_sizes=[2, 3, 4, 5],
        char_vocab_size=len(char_tokenizer.word_index)+1, chars_len=CHARS_LEN,
        pos_vocab_size=len(pos_tokenizer.word_index)+1, pos_emb_dim=16, pos_hid_dim=128, pos_len=WORDS_LEN,
        condition_num=21)

    # pretrain generator
    pretrain_generator(generator, Ws[0], Ws[1], As[0], As[1],
                       word_tokenizer, encoder, word_decoder, attention)

    # pretrain discriminator
    pretrain_discriminator(discriminator, Cs[0], Cs[1], Ps[0], Ps[1], As[0], As[1],
                           char_tokenizer, pos_tokenizer)


if __name__ == "__main__":
    main()
