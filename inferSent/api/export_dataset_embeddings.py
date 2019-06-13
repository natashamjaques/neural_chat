""" Use pre-trained inferSent model and export sentence embeddings. """

import os
import pickle
import torch
from inferSent.encoder.models import InferSent
import nltk
import argparse
nltk.download('punkt')


def load_infersent_model(model_path, bsize=64, word_emb_dim=300, enc_lstm_dim=2048, version=1):
    params_model = {'bsize': bsize, 'word_emb_dim': word_emb_dim, 'enc_lstm_dim': enc_lstm_dim,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    infersent_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

    if version == 1:
        W2V_PATH = os.path.join(os.path.join(infersent_dataset_dir, 'GloVe'), 'glove.840B.300d.txt')
    elif version == 2:
        W2V_PATH = os.path.join(os.path.join(infersent_dataset_dir, 'fastText'), 'crawl-300d-2M-subword.vec')
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)
    return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filepath', type=str,
                           required=False, help="Dataset path")
    argparser.add_argument('--streaming', action="store_true", default=False, help="Whether to use streaming "
                                                                                   "pickle or dump everything "
                                                                                   "at once. Use streaming for "
                                                                                   "large datasets")
    argparser.add_argument('--debuglen', type=int, default=5, help="Number of the sentences to show the output for")
    argparser.add_argument('--step', type=int, default=100, help="Number of steps for saving output")
    argparser.add_argument('--version', type=int, default=1, help="Which model version of inferSent to use. "
                                                                  "V1 has been trained on GloVe. "
                                                                  "V2 has been trained on fastText.")
    argparser.add_argument('--bsize', type=int, default=64, help="batch size")
    argparser.add_argument('--word_emb_dim', type=int, default=300, help="Dimension of word embeddings")
    argparser.add_argument('--enc_lstm_dim', type=int, default=2048, help="Dimension of LSTM")

    args = argparser.parse_args()
    infersent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(os.path.join(infersent_dir, 'encoder'), f'infersent{args.version}.pickle')
    model = load_infersent_model(MODEL_PATH, bsize=args.bsize, word_emb_dim=args.word_emb_dim,
                                 enc_lstm_dim=args.enc_lstm_dim, version=args.version)

    sentence_embeddings = []
    prefix = os.path.dirname(args.filepath)
    if args.streaming:
        output_path = os.path.join(prefix, f'sentence_embeddings_{args.version}_streaming.pkl')
    else:
        output_path = os.path.join(prefix, f'sentence_embeddings_{args.version}.pkl')

    sentences = pickle.load(open(args.filepath, 'rb'))
    flattened_sentences = [utterance for conversation in sentences for utterance in conversation]

    print('Encoding sentences ...')
    flattened_embeddings = model.encode(flattened_sentences, tokenize=True, bsize=64)
    print('InferSent encoding done.')
    idx = 0
    sent_idx = 0
    for conversation in sentences:
        idx += 1
        conversation_embeddings = []
        for sentence in conversation:
            conversation_embeddings += [list(flattened_embeddings[sent_idx])]
            if idx < args.debuglen:
                print(flattened_embeddings[sent_idx])
            sent_idx += 1
        if idx % args.step == 0:
            print(f'Conversations: {idx}, Sentences embedded: {sent_idx}')
        if args.streaming:
            if idx == 1:
                pickle.dump(conversation_embeddings, open(output_path, 'wb'))
            else:
                pickle.dump(conversation_embeddings, open(output_path, 'ab'))
        sentence_embeddings += [conversation_embeddings]
    print(f'Conversations: {idx}, Sentences embedded: {sent_idx}')
    if not args.streaming:
        pickle.dump(sentence_embeddings, open(output_path, 'wb'))

