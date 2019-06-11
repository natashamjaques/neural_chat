import argparse
import os
import sys
import pandas as pd
import re
import pickle

from model.solver import Solver, VariationalSolver
from model.data_loader import get_loader
from model.configs import get_config_from_dir
from model.utils import Vocab, Tokenizer
from model.models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test') # or valid
    parser.add_argument('--results_df_path', type=str, default=None,
                        help='Path to a csv dataframe to store results')
    kwargs = parser.parse_args()

    config = get_config_from_dir(kwargs.checkpoint, mode=kwargs.mode)

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
        shuffle=False,
        emojis=emotion_sentences,
        infersent=infersent_sentences)

    if config.model in VariationalModels:
        solver = VariationalSolver(config, None, data_loader, vocab=vocab, is_train=False)
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)

    solver.build()
    average, extrema, greedy = solver.embedding_metric()

    if kwargs.results_df_path is not None:
        row = {'checkpoint': kwargs.checkpoint, 'dataset': kwargs.dataset,
               'data': config.data, 'embedding_average': average,
               'embedding_extreme': extrema, 'embedding_greedy': greedy,
               'emotion': config.emotion, 'infersent': config.infersent,
               'input_only': config.context_input_only}
        if os.path.exists(kwargs.results_df_path):
            df = pd.read_csv(kwargs.results_df_path)
        else:
            df = pd.DataFrame()
        df = df.append(row, ignore_index=True)
        df.to_csv(kwargs.results_df_path)