from solver import *
from data_loader import get_loader
from configs import get_config
from model.utils import Vocab
import os
import pickle
from model.models import VariationalModels


def load_pickle(path):
    if 'streaming' in str(path):
        return load_streaming_pickle(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_streaming_pickle(path):
    items = []
    with open(path, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            items += [item]
    return items


if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    # Load emojis if necessary
    emotion_sentences = None
    val_emotion_sentences = None
    if config.emotion:
        print('Loading emojis...')
        emotion_sentences = load_pickle(config.emojis_path)
        val_emotion_sentences = load_pickle(val_config.emojis_path)

    # Load infersent embeddings if necessary
    infersent_sentences = None
    val_infersent_sentences = None
    if config.infersent:
        print('Loading infersent sentence embeddings...')
        infersent_sentences = load_pickle(config.infersent_path)
        val_infersent_sentences = load_pickle(val_config.infersent_path)
        embedding_size = infersent_sentences[0][0].shape[0]
        config.infersent_output_size = embedding_size

    train_data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size,
        emojis=emotion_sentences,
        infersent=infersent_sentences)

    eval_data_loader = get_loader(
        sentences=load_pickle(val_config.sentences_path),
        conversation_length=load_pickle(val_config.conversation_length_path),
        sentence_length=load_pickle(val_config.sentence_length_path),
        vocab=vocab,
        batch_size=val_config.eval_batch_size,
        shuffle=False,
        emojis=val_emotion_sentences,
        infersent=val_infersent_sentences)

    # for testing
    # train_data_loader = eval_data_loader
    if config.model in VariationalModels:
        solver = VariationalSolver
    else:
        solver = Solver

    solver = solver(config, train_data_loader, eval_data_loader, vocab=vocab, 
                    is_train=True)

    solver.build()
    solver.train()
