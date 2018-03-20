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


def generate_sequence(encoder, word_decoder, attention, word_tokenizer,
                      texts=["私は猫です。", "僕は犬です。"], authors=["夏目漱石", "森鴎外"], max_len=50):

    bos_eos = word_tokenizer.texts_to_sequences(["<s>", "</s>"])

    for X, Y in batch_generator(pd.Series(texts), pd.Series(authors),
        word_tokenizer, batch_size=len(texts), max_len=max_len):
        break

    results = []        
    for text, author, x, y in zip(texts, authors, X, Y):
        x = np.reshape(x, (1, -1))
        y = np.reshape(y, (1, -1))

        target_seq = np.array(bos_eos[0])
        output_seq = bos_eos[0][:]
        attention_seq = np.empty((0,max_len))
        prev_token_index = bos_eos[0][:]  # 1つ前のトークン

        encoded_seq, *states_value = encoder.predict(x)

        while True:
            decoded_seq, *states_value = word_decoder.predict([target_seq] + states_value)
            output_tokens, attention = attention.predict([encoded_seq, decoded_seq, y])  # condition
            sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
            
            if prev_token_index == sampled_token_index:
                break
            
            output_seq += sampled_token_index
            attention_seq = np.append(attention_seq, attention[0], axis=0)

            if (sampled_token_index == bos_eos[1] or len(output_seq) > max_len):
                break

            target_seq = np.array(sampled_token_index)
            prev_token_index = sampled_token_index

        results.append(
            (
                text, author,
                ''.join(word_tokenizer.texts_to_sequences(output_seq, return_words=True))
            )
        )
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
        generate_sequence(self._encoder, self._woed_decoder, self._attention, self._tokenizer)


def pretrain_generator(generator, train_C, test_C, train_A, test_A, word_tokenizer, encoder, word_decoder, attention):
    """GeneratorのTraining
    """
    generator.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy')

    checkpoint_cb = ModelCheckpoint("models/pretrain_generator", period=10)
    simple_test_cb = PretrainGeneratorCallBack(
        encoder, word_decoder, attention, word_tokenizer)

    generator.fit_generator(
        generator=pretrain_batch_generator(train_C, train_A, word_tokenizer),
        steps_per_epoch=100,
        epochs=100, verbose=2,
        validation_data=pretrain_batch_generator(
            test_C, test_A, word_tokenizer),
        validation_steps=1,
        callbacks=[checkpoint_cb, simple_test_cb]
    )


def main():
    aozora = "./data/sample_aozora.csv"
    wikipedia = "./data/sample_wikipedia.csv"

    Cs, Ws, Ps, As, tokenizers = load_dataset(aozora, wikipedia)
    char_tokenizer, word_tokenizer, pos_tokenizer = tokenizers

    # initialize
    encoder, generator, word_decoder, attention = get_generator(
        vocab_size=len(char_tokenizer.word_index), emb_dim=512, hid_dim=1024, att_dim=1024, condition_num=21, max_word=50
    )
    discriminator = get_discriminator()

    # pretrain generator
    pretrain_generator(generator, Ws[0], Ws[1], As[0], As[1], word_tokenizer, encoder, word_decoder, attention)


if __name__ == "__main__":
    main()
