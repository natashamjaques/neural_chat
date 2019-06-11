import argparse
from solver import Solver, VariationalSolver
from data_loader import get_loader
from model.configs import get_config_from_dir
from model.utils import Vocab, Tokenizer
import os
import pickle
from model.models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test')  # or valid
    kwargs = parser.parse_args()

    config = get_config_from_dir(kwargs.checkpoint, mode=kwargs.mode)
    print(config)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    emotion_sentences = None
    if config.emotion:
        emotion_sentences = load_pickle(config.emojis_path)

    # Load infersent embeddings if necessary
    infersent_sentences = None
    if config.infersent:
        print('Loading infersent sentence embeddings...')
        infersent_sentences = load_pickle(config.infersent_path)
        embedding_size = infersent_sentences[0][0].shape[0]
        config.infersent_output_size = embedding_size

    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size,
        emojis=emotion_sentences,
        infersent=infersent_sentences)

    if config.model in VariationalModels:
        solver = VariationalSolver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        solver.importance_sample()
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        solver.test()