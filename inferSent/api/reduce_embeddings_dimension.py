""" Use saved inferSent sentence embeddings and reduce their dimensionality using Linear PCA.
For each dataset, the training subset is used for building the PCA transformation for all
train, validation, and test sets.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import argparse


def load_pickle(path):
    if 'streaming' in path:
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


def fit_pca(inp, num_components):
    # 0 < n_components < 1 represents explained variance,
    # ow it is the fixed number of components
    pca = PCA(n_components=num_components)
    pca_fit = pca.fit(inp)
    return pca_fit


def transform_with_pca(pca_fit, inp):
    transformed = pca_fit.transform(inp)
    print(np.shape(transformed))
    return transformed


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str,
                           required=False, help="Dataset path")
    argparser.add_argument('--fixeddim', action="store_true", default=False,
                           help="Whether to have a fixed number of ouput dimensions using args.ndim or "
                                "to infer that with args.explainedvar")
    argparser.add_argument('--savepca', action="store_true", default=False,
                           help="Whether to save the constructed training pca.")
    argparser.add_argument('--exportembeddings', action="store_true", default=False,
                           help="Whether to export PCA reduced embeddings.")
    argparser.add_argument('--ndim', type=int, default=450, help="Number of the PCA output dimensions")
    argparser.add_argument('--explainedvar', type=float, default=0.95, help="Explained variance")
    argparser.add_argument('--debuglen', type=int, default=5, help="Number of the sentences to show the output for")
    argparser.add_argument('--step', type=int, default=100, help="Number of steps for saving output")
    argparser.add_argument('--version', type=int, default=1, help="Which model version of inferSent to use. "
                                                                  "V1 has been trained on GloVe. "
                                                                  "V2 has been trained on fastText.")
    args = argparser.parse_args()

    train_dir = os.path.join(args.dataset, 'train')
    train_file_path = os.path.join(train_dir, f'sentence_embeddings_{args.version}_streaming.pkl')
    train_embeddings = load_pickle(train_file_path)
    flattened_train_embeddings = [utterance for conversation in train_embeddings for utterance in conversation]

    if args.fixeddim:
        pca_model_file_path = os.path.join(train_dir, f'v{args.version}_PCA_model_{args.ndim}.pkl')
        if args.savepca:
            pca_embeddings = fit_pca(flattened_train_embeddings, args.ndim)
            pickle.dump(pca_embeddings, open(pca_model_file_path, 'wb'))
        else:
            pca_embeddings = pickle.load(open(pca_model_file_path, 'rb'))
    else:
        pca_model_file_path = os.path.join(train_dir, f'v{args.version}_PCA_model_{args.explainedvar}.pkl')
        if args.savepca:
            pca_embeddings = fit_pca(flattened_train_embeddings, args.explainedvar)
            pickle.dump(pca_embeddings, open(pca_model_file_path, 'wb'))
        else:
            pca_embeddings = pickle.load(open(pca_model_file_path, 'rb'))

    if args.exportembeddings:
        datasets = ['train', 'valid', 'test']
        for dataset in datasets:
            dataset_dir = os.path.join(args.dataset, dataset)
            if args.fixeddim:
                output_path = os.path.join(dataset_dir, f'sentence_embeddings_{args.version}_PCA_{args.ndim}.pkl')
            else:
                output_path = os.path.join(dataset_dir,
                                           f'sentence_embeddings_{args.version}_PCA_{args.explainedvar}.pkl')
            if dataset == 'train':
                embeddings = train_embeddings
                flattened_embeddings = flattened_train_embeddings
            else:
                file_path = os.path.join(dataset_dir, f'sentence_embeddings_{args.version}.pkl')
                embeddings = load_pickle(file_path)
                flattened_embeddings = [utterance for conversation in embeddings for utterance in conversation]
            transformed_embeddings = transform_with_pca(pca_embeddings, flattened_embeddings)

            idx = 0
            sent_idx = 0
            all_reduced = []
            for conversation in embeddings:
                idx += 1
                conv_reduced = []
                for sentence in conversation:
                    conv_reduced += [transformed_embeddings[sent_idx]]
                    if idx < args.debuglen:
                        print(np.shape(transformed_embeddings[sent_idx]))
                    sent_idx += 1
                if idx % args.step == 0:
                    print(f'{idx} Conversations, including {sent_idx} sentence embeddings reduced.')
                all_reduced += [conv_reduced]
            print(f'{idx} Conversations, including {sent_idx} sentence embeddings reduced. All done!')
            pickle.dump(all_reduced, open(output_path, 'wb'))
