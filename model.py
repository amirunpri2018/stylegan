# coding: utf-8

from keras import regularizers
from keras.layers import (LSTM, BatchNormalization, Bidirectional, Concatenate,
                          Convolution2D, Dense, Dropout, Embedding, Input,
                          MaxPooling2D, Reshape)
from keras.models import Model


def get_encoder(vocab_size=30000, emb_dim=128, hid_dim=128, condition_num=21, max_words=100):
    """分かち書きのEncoder

    分かち書きを入力として、エンコーディングする。
    それに作家の条件をつけて分かち書きの状態でデコードする。
    入力する作家は、青空文庫の20作家とwikipediaである。


    Returns:
        pretrain_generator
        generator
        encoder_model
        decoder_model -- 使わないかもしれないが、デコーダー単体。隠れ状態の初期値。
    """
    # Encoder
    encoder_input = Input(shape=(max_words,))  # 最大入力単語数100
    x = Embedding(vocab_size, emb_dim, mask_zero=True)(encoder_input)
    x = Bidirectional(
        LSTM(hid_dim, return_sequences=True, activation='relu'))(x)
    x = Dropout(0.5)(x)
    _, *encoder_states = Bidirectional(LSTM(hid_dim, return_state=True))(x)

    encoder_model = Model(inputs=[encoder_input], outputs=encoder_states)

    # Decoder
    # Inputs
    decoder_initial_states_inputs = [
        Input(shape=(hid_dim,)), Input(shape=(hid_dim,)),
        Input(shape=(hid_dim,)), Input(shape=(hid_dim,))
    ]
    decoder_condition_input = Input(
        shape=(condition_num,))  # 20人の作家 + wikipedia
    decoder_input = Input(shape=(max_words,))

    # 生成時にも利用する層の定義
    decoder_embedding = Embedding(vocab_size, emb_dim)
    decoder_lstm1 = LSTM(hid_dim, return_sequences=True, return_state=True)
    decoder_lstm2 = LSTM(hid_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(vocab_size, use_bias=True, activation='softmax')

    hidden_states = Concatenate()(decoder_initial_states_inputs +
                                  [decoder_condition_input])
    h_state = Dense(hid_dim, activation='relu')(hidden_states)
    c_state = Dense(hid_dim, activation='relu')(hidden_states)

    x = decoder_embedding(decoder_input)
    x, _, _ = decoder_lstm1(x, initial_state=[h_state, c_state])
    x = Dropout(0.5)(x)
    x, _, _ = decoder_lstm2(x, initial_state=[h_state, c_state])
    x = Dropout(0.5)(x)
    decoder_outputs = decoder_dense(x)

    decoder_model = Model(
        decoder_initial_states_inputs +
        [decoder_condition_input, decoder_input],
        decoder_outputs
    )

    # pretrain用モデル
    pretrain_generator = Model(
        encoder_model.inputs + [decoder_condition_input, decoder_input],
        decoder_model(encoder_model.outputs +
                      [decoder_condition_input, decoder_input])
    )

    # 本格的に訓練されるgenerator
    generator_input = Input(shape=(1,))  # 1単語入力
    generator_condition_input = Input(shape=(condition_num, ))  # 作家
    generator_states1_inputs = [Input(shape=(hid_dim,)), Input(
        shape=(hid_dim,))]  # decoder_lstm1の隠れ層
    generator_states2_inputs = [Input(shape=(hid_dim,)), Input(
        shape=(hid_dim,))]  # decoder_lstm2の隠れ層

    x = decoder_embedding(generator_input)
    x, *hidden_states1 = decoder_lstm1(x,
                                       initial_state=generator_states1_inputs)
    x = Dropout(0.5)(x)
    x, *hidden_states2 = decoder_lstm2(x,
                                       initial_state=generator_states2_inputs)
    x = Dropout(0.5)(x)
    generator_output = decoder_dense(x)

    generator = Model(
        [generator_input, generator_condition_input] +
        generator_states1_inputs + generator_states2_inputs,
        [generator_output] + hidden_states1 + hidden_states2
    )

    return pretrain_generator, generator, encoder_model, decoder_model


# seq_encoder_decoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
def get_discriminator(
        char_emb_dim=128, num_filters=64, filter_sizes=[2, 3, 4, 5], char_vocab_size=3000, max_chars=200, word_vocab_size=30000, word_emb_dim=128, word_hid_dim=128, max_words=100,
        pos_vocab_size=100, pos_emb_dim=16, pos_hid_dim=128, 
        condition_num=21):
    """複数の種類のDiscriminatorを組み合わせたDiscriminator。
    分類するラベルは、true ,falseである。

    - 文字レベル：Generatorの出力を文字レベルにした上でのCNN。
    - 分かち書きレベル：Generatorの出力をそのまま利用してRNN。
    - 品詞レベル：Generatorの出力を結合し、MeCabで形態素解析を行い、品詞系列にしてRNN

    以上の3つのレベルのDiscriminatorは最終的に結合され、2のクラスへの確率が出力され、Generatorに対する報酬になる。
    """

    # Character Level CNN
    char_input = Input(shape=(max_chars,))
    char_emb = Embedding(char_vocab_size, char_emb_dim)(char_input)
    char_emb = Reshape((max_chars, char_emb_dim, 1))(char_emb) 
    convs = []
    for filter_size in filter_sizes:
        x = Convolution2D(
            filters=num_filters, kernel_size=(filter_size, char_emb_dim),
            kernel_regularizer=regularizers.l2(0.001),
            activation="relu")(char_emb)
        x = Dropout(0.5)(x)
        x = MaxPooling2D(pool_size=(max_chars - filter_size + 1, 1))(x)
        x = Reshape((num_filters,))(x)
        convs.append(x)
    char_cnn = Concatenate()(convs)

    # wakati Level RNN
    wakati_input = Input(shape=(max_words,))  # 最大入力単語数100
    x = Embedding(word_vocab_size, word_emb_dim, mask_zero=True)(wakati_input)
    x = Bidirectional(
        LSTM(word_hid_dim, return_sequences=True, activation='relu'))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(word_hid_dim, return_sequences=True, activation='relu'))(x)
    x = Dropout(0.5)(x)
    _, *wakati_states = Bidirectional(LSTM(word_hid_dim, return_state=True))(x)
    wakati_rnn = Concatenate()(wakati_states)

    # POS Level RNN
    pos_input = Input(shape=(max_words,))  # 最大入力単語数
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
    x = Concatenate()([char_cnn, wakati_rnn, pos_rnn, condition_input])
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    discriminator = Model(
        inputs=[char_input, wakati_input, pos_input, condition_input],
        outputs=[output]
    )

    return discriminator
