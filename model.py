# coding: utf-8

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, LSTM, Bidirectional, Dense, Concatenate

def get_encoder(vocab_size=30000, emb_dim = 128, hid_dim = 128, condition_num=21, max_words=100):
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
    x = Bidirectional(LSTM(hid_dim, return_sequences=True, activation='relu'))(x)
    x = Dropout(0.5)(x)
    _, *encoder_states = Bidirectional(LSTM(hid_dim, return_state=True))(x)

    encoder_model = Model(inputs=[encoder_input], outputs=encoder_states)
    
    # Decoder
    # Inputs
    decoder_initial_states_inputs = [
        Input(shape=(hid_dim,)), Input(shape=(hid_dim,)),
        Input(shape=(hid_dim,)), Input(shape=(hid_dim,))
    ]
    decoder_condition_input = Input(shape=(condition_num,))  # 20人の作家 + wikipedia
    decoder_input = Input(shape=(max_words,))

    # 生成時にも利用する層の定義
    decoder_embedding = Embedding(vocab_size, emb_dim)
    decoder_lstm1 = LSTM(hid_dim, return_sequences=True, return_state=True)
    decoder_lstm2 = LSTM(hid_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(vocab_size, use_bias=True, activation='softmax')

    hidden_states = Concatenate()(decoder_initial_states_inputs + [decoder_condition_input])
    h_state = Dense(hid_dim, activation='relu')(hidden_states) 
    c_state = Dense(hid_dim, activation='relu')(hidden_states)

    x = decoder_embedding(decoder_input)
    x, _, _ = decoder_lstm1(x, initial_state=[h_state, c_state])
    x = Dropout(0.5)(x)
    x, _, _ = decoder_lstm2(x, initial_state=[h_state, c_state])
    x = Dropout(0.5)(x)
    decoder_outputs = decoder_dense(x)

    decoder_model = Model(
        decoder_initial_states_inputs + [decoder_condition_input, decoder_input],
        decoder_outputs
    )

    # pretrain用モデル
    pretrain_generator = Model(
        encoder_model.inputs + [decoder_condition_input, decoder_input],
        decoder_model(encoder_model.outputs + [decoder_condition_input, decoder_input])
    )
    
    # 本格的に訓練されるgenerator
    generator_input = Input(shape=(1,))  # 1単語入力
    generator_condition_input = Input(shape=(condition_num, ))  # 作家
    generator_states1_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]  # decoder_lstm1の隠れ層
    generator_states2_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]  # decoder_lstm2の隠れ層
    
    x = decoder_embedding(generator_input)
    x, *hidden_states1 = decoder_lstm1(x, initial_state=generator_states1_inputs)
    x = Dropout(0.5)(x)
    x, *hidden_states2 = decoder_lstm2(x, initial_state=generator_states2_inputs)
    x = Dropout(0.5)(x)
    generator_output = decoder_dense(x)

    generator = Model(
        [generator_input, generator_condition_input] + generator_states1_inputs + generator_states2_inputs,
        [generator_output] + hidden_states1 + hidden_states2
    )

    return pretrain_generator, generator, encoder_model, decoder_model



# seq_encoder_decoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
def get_discriminator():
    """複数の種類のDiscriminatorを組み合わせたDiscriminator。
    分類するラベルは、各作家、wikipedia、falseである。

    - 文字レベル：Generatorの出力を文字レベルにした上で分類する。
    - 分かち書きレベル：Generatorの出力をそのまま利用して分類する。
    - 品詞レベル：Generatorの出力を結合し、MeCabで形態素解析を行い、品詞系列にして分類する。

    以上の3つのレベルのDiscriminatorは最終的に結合され、22のクラスへの確率が出力され、Generatorに対する報酬になる。
    """
    pass