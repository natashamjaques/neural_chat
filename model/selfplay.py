import argparse
import csv

from solver import Solver, VariationalSolver
from data_loader import get_loader
from configs import get_config_from_dir
from model.utils import Vocab, Tokenizer
import os
import pickle
from model.models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


greetings = [
    'Good morning',
    'Hello',
    'Hi',
    'How are you?',
    'What\'s up?',
    'Wudup',
    'Hey!',
    'Everything okay?',
    'Tell me about you',
    'What\'s your story?',
    'Having fun?',
    'What are you doing this weekend?',
    'Long time no see!',
    'Greetings',
    'Where have you been?',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=5)
    parser.add_argument('--conversation_length', type=int, default=10)
    parser.add_argument('--print-history', action='store_true')
    parser.add_argument('--kind', type=str, default='toxic', choices=['toxic', 'notoxic'])

    # Allow sampling with higher priority to samples with longer length and
    # fewer unknowns
    parser.add_argument('--sample_by', type=str, default='priority',
                        help="Type of sampling for generated responses. Can" + \
                             "be None, 'priority', or 'length'")

    # Turn on debugging outputs
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)

    # For interacting with rl models
    parser.add_argument('--load_rl_ckpt', action='store_true')
    parser.set_defaults(load_rl_ckpt=False)
    parser.add_argument('--rl_ckpt_epoch', type=int, default=None)

    kwargs = parser.parse_args()
    kw_dict = vars(kwargs)
    kw_dict['mode'] = 'test'

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
    with open(f'/code/datasets/conversations_{kwargs.kind}.csv', 'w') as f:
        writer = csv.writer(f)
        for i in greetings:
            conv = solver.self_play(conversation_length=kwargs.conversation_length,
                                    max_sentence_length=kwargs.max_sentence_length,
                                    max_conversation_length=kwargs.max_conversation_length,
                                    sample_by=kwargs.sample_by,
                                    print_history=kwargs.print_history,
                                    first_interaction=i)
            writer.writerows([[c] for c in conv])
