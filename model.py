# coding: utf-8

from keras import regularizers
from keras.layers import (LSTM, BatchNormalization, Bidirectional, Concatenate,
                          Convolution2D, Dense, Dropout, Embedding, Input,
                          MaxPooling2D, Reshape, Activation, dot, concatenate)
from keras.models import Model


def get_generator(vocab_size=1000, emb_dim=128, hid_dim=128, att_dim=1024, condition_num=21, words_len=100):
    """分かち書きのGenerator

    分かち書きを入力として、エンコーディングする。
    それに作家の条件をつけて分かち書きの状態でデコードする。
    入力する作家は、青空文庫の20作家とwikipediaである。

    Returns:
        generator
        encoder
        word_decoder
    """
    # Embeddeingは共通
    embedding = Embedding(vocab_size, emb_dim, mask_zero=True)

    # Encoder
    encoder_inputs = Input(shape=(words_len,))
    encoder_embedded = embedding(encoder_inputs)
    encoded_seq, *encoder_states = Bidirectional(LSTM(hid_dim, return_sequences=True, return_state=True))(encoder_embedded)
    encoder_states_h = Concatenate()(encoder_states[:2])
    encoder_states_c = Concatenate()(encoder_states[2:])
    
    encoder_states_h = Dense(hid_dim, activation='relu',  kernel_initializer='he_normal')(encoder_states_h)
    encoder_states_c = Dense(hid_dim, activation='relu',  kernel_initializer='he_normal')(encoder_states_c)
    encoder_states = [encoder_states_h, encoder_states_c]

    # Encoder Model
    encoder = Model(encoder_inputs, [encoded_seq] + encoder_states)

    # デコーダー
    decoder_inputs = Input(shape=(words_len,))
    decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)

    decoder_embedded = embedding(decoder_inputs)
    decoded_seq, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

    # Attention
    condition_input = Input(shape=(words_len, condition_num))
    score_dense = Dense(hid_dim*2,  kernel_initializer='he_normal')
    attention_dense = Dense(att_dim, activation='tanh', kernel_initializer='he_normal')
    output_dense = Dense(vocab_size, activation='softmax',  kernel_initializer='he_normal')

    score = score_dense(decoded_seq)
    score = dot([score, encoded_seq], axes=(2,2))
    attention = Activation('softmax')(score)
    context = dot([attention, encoded_seq], axes=(2,1))     
    # ここでのみ条件が使われる
    concat = concatenate([context, decoded_seq, condition_input])  # axis=2
    attentional = attention_dense(concat)
    outputs = output_dense(attentional)

    # Generator Model
    generator = Model([encoder_inputs, decoder_inputs, condition_input], outputs)

    
    # Decoder
    word_decoder_inputs = Input(shape=(1,))
    word_decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]

    word_decoder_embeded = embedding(word_decoder_inputs)
    word_decoded_seq, *word_decoder_states = decoder_lstm(word_decoder_embeded, initial_state=word_decoder_states_inputs)

    word_decoder = Model(
        inputs=[word_decoder_inputs] + word_decoder_states_inputs,
        outputs=[word_decoded_seq] + word_decoder_states
    )

    # Attention
    encoded_seq_in = Input(shape=(words_len, hid_dim*2))
    decoded_seq_in = Input(shape=(1, hid_dim))
    condition_in = Input(shape=(1, condition_num))

    score = score_dense(decoded_seq_in)
    score = dot([score, encoded_seq_in], axes=(2,2))
    attention = Activation('softmax')(score)
    
    context = dot([attention, encoded_seq_in], axes=(2,1))
    concat = concatenate([context, decoded_seq_in, condition_in])
    attentional = attention_dense(concat)
    attention_outputs = output_dense(attentional)
    
    attention_model = Model([encoded_seq_in, decoded_seq_in, condition_in], [attention_outputs, attention])


    return encoder, generator, word_decoder, attention_model


def get_discriminator(
        char_emb_dim=128, num_filters=64, filter_sizes=[2, 3, 4, 5], char_vocab_size=3000, chars_len=200,
        pos_vocab_size=100, pos_emb_dim=16, pos_hid_dim=128, pos_len=100,
        condition_num=21):
    """複数の種類のDiscriminatorを組み合わせたDiscriminator。
    分類するラベルは、true ,falseである。

    - 文字レベル：Generatorの出力を文字レベルにした上でのCNN。
    - 品詞レベル：Generatorの出力を結合し、MeCabで形態素解析を行い、品詞系列にしてRNN

    以上の2つのレベルのDiscriminatorは最終的に結合され、2のクラスへの確率が出力され、Generatorに対する報酬になる。
    """

    # Character Level CNN
    char_input = Input(shape=(chars_len,))
    char_emb = Embedding(char_vocab_size, char_emb_dim)(char_input)
    char_emb = Reshape((chars_len, char_emb_dim, 1))(char_emb)
    convs = []
    for filter_size in filter_sizes:
        x = Convolution2D(
            filters=num_filters, kernel_size=(filter_size, char_emb_dim),
            kernel_regularizer=regularizers.l2(0.001),
            activation="relu")(char_emb)
        x = Dropout(0.5)(x)
        x = MaxPooling2D(pool_size=(chars_len - filter_size + 1, 1))(x)
        x = Reshape((num_filters,))(x)
        convs.append(x)
    char_cnn = Concatenate()(convs)

    # POS Level RNN
    pos_input = Input(shape=(pos_len,))  # 最大入力単語数
    x = Embedding(pos_vocab_size, pos_emb_dim, mask_zero=True)(pos_input)
    x = Bidirectional(
        LSTM(pos_hid_dim, return_sequences=True, activation='relu'))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(pos_hid_dim, return_sequences=True, activation='relu'))(x)
    x = Dropout(0.5)(x)
    _, *pos_states = Bidirectional(LSTM(pos_hid_dim, return_state=True))(x)
    pos_rnn = Concatenate()(pos_states)

    # condition input and judge
    condition_input = Input(shape=(condition_num,))
    x = Concatenate()([char_cnn, pos_rnn, condition_input])
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    discriminator = Model(
        inputs=[char_input, pos_input, condition_input],
        outputs=[output]
    )

    return discriminator


def change_trainable(model, trainable=True):
    pass

