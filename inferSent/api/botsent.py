# -*- coding: utf-8 -*-

""" Export InferSent embedding or its PCA reduced version for a given text/set of texts. """

from __future__ import print_function, division, unicode_literals
import pickle
import nltk
import os
import argparse
nltk.download('punkt')


class Botsent():
    def __init__(self, pretrained_path, version=1, explained_var=0.95, 
                 use_pca=True):
        self.pretrained_path = str(pretrained_path)
        self.version = version
        self.explained_var = explained_var
        self.infersent_model_path = os.path.join(
            self.pretrained_path,  f'updated_infersent{self.version}_converted.pkl')
        self.model = pickle.load(open(self.infersent_model_path, 'rb'))
        self.use_pca = use_pca
        if use_pca:
            self.pca_path = os.path.join(
                self.pretrained_path, 
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
    argparser.add_argument('--pretrainedpath', type=str,
                           help="Link to the dataset that has the saved pretrained infersent and pca model.")
    argparser.add_argument('--version', type=int, default=1, help="Which model version of inferSent to use. "
                                                                  "V1 has been trained on GloVe. "
                                                                  "V2 has been trained on fastText.")
    argparser.add_argument('--explainedvar', type=float, default=0.95, help="Explained variance")
    argparser.add_argument('--text', type=str, required=True, help="Input text to infer embedding on")
    argparser.add_argument('--maxlen', type=int, default=30, help="Max length of input text")
    args = argparser.parse_args()

    botsent = Botsent(pretrained_path=args.pretrainedpath, version=args.version,
                      explained_var=args.explainedvar)
    embedding = botsent.encode(args.text)
    print(args.text, embedding)
