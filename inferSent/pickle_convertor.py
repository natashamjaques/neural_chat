# -*- coding: utf-8 -*-

""" Use inferSent and a saved PCA reduction to predict a compact sentence embedding
 from a single text input
"""

from __future__ import print_function, division, unicode_literals
import pickle
import nltk
import os
import argparse
nltk.download('punkt')

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
print(sys.path)

# This file saves a converted version of the infersent model files

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pretrainedpath', type=str,
                           help="Link to the dataset that has the saved pretrained infersent and pca model.")
    argparser.add_argument('--version', type=int, default=1, help="Which model version of inferSent to use. "
                                                                  "V1 has been trained on GloVe. "
                                                                  "V2 has been trained on fastText.")
    argparser.add_argument('--explainedvar', type=float, default=0.95, help="Explained variance")
    args = argparser.parse_args()

    from botsent import Botsent
    botsent = Botsent(pretrained_path=args.pretrainedpath, version=args.version,
                      explained_var=args.explainedvar)

    from inferSent.models import InferSent
    converted_model = InferSent()
    converted_model.copy(botsent.model)
    pickle.dump(converted_model, open(botsent.infersent_model_path[0:-4] + '_converted.pkl', 'wb'))

