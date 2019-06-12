# -*- coding: utf-8 -*-

""" Export InferSent embedding or its PCA reduced version for a given text/set of texts. """

from __future__ import print_function, division, unicode_literals
from inferSent.api.export_dataset_embeddings import load_infersent_model
import pickle
import nltk
import os
import argparse


class Botsent():
    def __init__(self, train_set_path, version=1, explained_var=0.95,
                 use_pca=True):
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(str(train_set_path)))))
        infersent_path = os.path.join(os.path.join(root_path, 'inferSent'), 'encoder')
        self.version = version
        self.explained_var = explained_var
        self.infersent_model_path = os.path.join(infersent_path, f'infersent{self.version}.pickle')
        self.model = load_infersent_model(self.infersent_model_path)
        self.use_pca = use_pca
        if use_pca:
            self.pca_path = os.path.join(
                str(train_set_path),
                f'v{self.version}_PCA_model_{self.explained_var}.pkl')
            self.pca_fit = pickle.load(open(self.pca_path, 'rb'))

    def encode(self, text):
        embedding = self.model.encode([text], tokenize=True)
        if self.use_pca:
            transformed = self.pca_fit.transform(embedding)
            return transformed[0]
        else:
            return embedding

    def encode_multiple(self, texts, override=False, override_use_pca=None):
        filtered_texts = ['_' if text == '' else text for text in texts]
        embedding = self.model.encode(filtered_texts, tokenize=True)

        use_pca = self.use_pca
        if override:
            use_pca = override_use_pca
        if use_pca:
            transformed = self.pca_fit.transform(embedding)
            return transformed
        else:
            return embedding


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--trainsetpath', type=str,
                           help="Link to the training set. This is used to locate PCA model and pre-trained infersent.")
    argparser.add_argument('--version', type=int, default=1, help="Which model version of inferSent to use. "
                                                                  "V1 has been trained on GloVe. "
                                                                  "V2 has been trained on fastText.")
    argparser.add_argument('--explainedvar', type=float, default=0.95, help="Explained variance")
    argparser.add_argument('--text', type=str, required=True, help="Input text to infer embedding on")
    argparser.add_argument('--maxlen', type=int, default=30, help="Max length of input text")
    args = argparser.parse_args()

    botsent = Botsent(train_set_path=args.trainsetpath, version=args.version, explained_var=args.explainedvar)
    embedding = botsent.encode(args.text)
    print(args.text, embedding)
