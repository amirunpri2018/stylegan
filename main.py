# coding: utf-8

from model import get_generator, get_discriminator
from dataloader import pretrain_batch_generator

from dataloader import load_dataset

from nltk.translate.bleu_score import sentence_bleu


def train_pretrain_generator(pretrain_generator, train_W, test_W, train_A, test_A, char_tokenizer):
    """GeneratorのTraining
    """
    pretrain_generator.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    pretrain_generator.fit_generator(
        generator=pretrain_batch_generator(train_W, train_A, char_tokenizer),
        steps_per_epoch=1000,
        epochs=1000, verbose=2,
        validation_data=pretrain_batch_generator(test_W, test_A, char_tokenizer),
        validation_steps=1
    )


def main():
    aozora = "./data/sample_aozora.csv"
    wikipedia = "./data/sample_wikipedia.csv"

    Cs, Ps, As, tokenizers = load_dataset(aozora, wikipedia)
    char_tokenizer, pos_tokenizer = tokenizers

    # initialize
    pretrain_generator, generator = get_generator(
        vocab_size=len(char_tokenizer.word_index), emb_dim=128, hid_dim=128, condition_num=21, max_words=100
    )
    discriminator = get_discriminator()

    # pretrain generator
    train_pretrain_generator(pretrain_generator, Cs[0], Cs[1], As[0], As[1], char_tokenizer)

if __name__ == "__main__":
    main()
