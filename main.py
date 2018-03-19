# coding: utf-8

import numpy as np
import pandas as pd
from keras_tqdm import TQDMNotebookCallback

from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

from dataloader import load_dataset, pretrain_batch_generator, batch_generator
from model import get_discriminator, get_generator


def generate_sequence(encoder, generator, char_tokenizer,
                      texts=["私は猫です。", "僕は犬です。"], authors=["夏目漱石", "森鴎外"], max_len=200):

    bos_eos = char_tokenizer.texts_to_sequences(["<s>", "</s>"])

    results = []
    for X, Y in batch_generator(pd.Series(texts), pd.Series(authors), char_tokenizer, batch_size=len(texts), max_len=max_len):
        for text, author, x, y in zip(texts, authors, X, Y):
            target_seq = np.array(bos_eos[0])
            output_seq = bos_eos[0][:]

            x = np.reshape(x, (1, -1))
            y = np.reshape(y, (1, -1))

            states_value = encoder.predict([x, y])

            while True:
                output_tokens, *states_value = generator.predict(
                    [target_seq] + states_value
                )
                sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
                output_seq += sampled_token_index

                if (sampled_token_index == bos_eos[1] or len(output_seq) > 200):
                    break

            target_seq = np.array(sampled_token_index)
            results.append(
                (
                    text, author,
                    ''.join(char_tokenizer.texts_to_sequences(output_seq, return_words=True))
                )
            )
        return results


class PretrainGeneratorCallBack(Callback):

    def __init__(self, encoder, generator, char_tokenizer):
        self._encoder = encoder
        self._generator = generator
        self._char_tokenizer = char_tokenizer

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        generate_sequence(self._encoder, self._generator, self._char_tokenizer)


def train_pretrain_generator(pretrain_generator, train_C, test_C, train_A, test_A, char_tokenizer, encoder, generator):
    """GeneratorのTraining
    """
    pretrain_generator.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy')

    checkpoint_cb = ModelCheckpoint("models/pretrain_generator", period=10)
    simple_test_cb = PretrainGeneratorCallBack(
        encoder, generator, char_tokenizer)

    pretrain_generator.fit_generator(
        generator=pretrain_batch_generator(train_C, train_A, char_tokenizer),
        steps_per_epoch=100,
        epochs=100, verbose=2,
        validation_data=pretrain_batch_generator(
            test_C, test_A, char_tokenizer),
        validation_steps=1,
        callbacks=[checkpoint_cb, simple_test_cb]
    )


def main():
    aozora = "./data/sample_aozora.csv"
    wikipedia = "./data/sample_wikipedia.csv"

    Cs, Ps, As, tokenizers = load_dataset(aozora, wikipedia)
    char_tokenizer, pos_tokenizer = tokenizers

    # initialize
    pretrain_generator, generator, encoder = get_generator(
        vocab_size=len(char_tokenizer.word_index), emb_dim=128, hid_dim=128, condition_num=21, max_chars=200
    )
    discriminator = get_discriminator()

    # pretrain generator
    train_pretrain_generator(
        pretrain_generator, Cs[0], Cs[1], As[0], As[1], char_tokenizer, encoder, generator)


if __name__ == "__main__":
    main()
