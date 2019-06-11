# TODO
# sample n conversations with a bot, that stops at x length convo
# save to a buffer similar to what we have
import argparse
from model.solver import Solver, VariationalSolver
from model.configs import get_config_from_dir
from model.data_loader import get_loader
from model.utils import Vocab
import pickle
import pandas as pd
import numpy as np

from model.models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


BOT_DICT = {'cornell/hred_canonical': ('HRED_cornell', 0),
            'cornell/vhred_canonical': ('VHRED_cornell', 1),
            'cornell/vhcr_canonical': ('VHCR_cornell', 2),
            'cornell/emoinfer_hred_AG': ('emoinfer_HRED_cornell', 3),
            'cornell/emoinfer_vhred_hparam': ('emoinfer_VHRED_cornell', 4),
            'cornell/emoinfer_vhcr_AG': ('emoinfer_VHCR_cornell', 5),
            'reddit_casual/hred_new_hparams': ('HRED_reddit_casual', 6),
            'reddit_casual/vhred_new_hparams': ('VHRED_reddit_casual', 7),
            'reddit_casual/vhcr_new_hparams': ('VHCR_reddit_casual', 8),
            'reddit_casual/emoinfer_hred_AG': ('emoinfer_HRED_casual', 9),
            'reddit_casual/emoinfer_vhred_AG': ('emoinfer_VHRED_casual', 10),
            'reddit_casual/emoinfer_vhcr_AG': ('emoinfer_VHCR_casual', 11)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--max_sentence_length', type=int, default=30)
    parser.add_argument('--max_conversation_length', type=int, default=5)
    parser.add_argument('--sample_by', type=str, default='priority',
                        help="Type of sampling for generated responses. Can be None, 'priority', or 'length'")

    parser.add_argument('--conversation_length', type=int, default=10)
    parser.add_argument('--sample_conversations', type=int, default=100)

    kwargs = parser.parse_args()
    kw_dict = vars(kwargs)

    checkpoint_pieces = kwargs.checkpoint.split('/')
    if len(checkpoint_pieces) < 2:
        print('Checkpoint name does not follow expected format.')
    bot_checkpoint_name = checkpoint_pieces[-2] + '/' + checkpoint_pieces[-1]
    (cur_bot_id, cur_bot_ind) = BOT_DICT[bot_checkpoint_name]

    config = get_config_from_dir(kwargs.checkpoint, **kw_dict)
    config.beam_size = 5

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    # If checkpoint is for an emotion model, load that pickle file
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
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)

    solver.build()

    self_play_buffer = pd.DataFrame()

    for i in range(kwargs.sample_conversations):
        messages = solver.self_play(conversation_length=kwargs.conversation_length,
                                    max_sentence_length=kwargs.max_sentence_length,
                                    max_conversation_length=kwargs.max_conversation_length,
                                    sample_by=kwargs.sample_by)
        responses = messages[1:]+['']
        conv_df = pd.DataFrame()
        conv_df['Message'] = messages
        conv_df['Response'] = responses
        conv_df['Response Rating'] = 0
        conv_df['ID'] = len(self_play_buffer) + np.arange(len(messages))
        conv_df['Chat ID'] = i + kwargs.sample_conversations * cur_bot_ind
        conv_df['Chatbot ID'] = cur_bot_id
        conv_df['Datetime'] = len(self_play_buffer) + np.arange(len(messages))
        self_play_buffer = pd.concat([self_play_buffer, conv_df])
    self_play_buffer.reset_index(drop=True)
    self_play_buffer.to_csv(
        f'metrics_and_analysis/self_play/{cur_bot_id}_{kwargs.sample_conversations}.csv')
