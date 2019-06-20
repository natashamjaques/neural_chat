# -*- coding: utf-8 -*-

""" Use torchMoji to predict emojis from a given text/set of texts.  """

from __future__ import print_function, division, unicode_literals

from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_emojis
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from torchMoji.api.dataset_emojize import EMOJIS

import json
import argparse

import numpy as np
import emoji


def top_emojis(array, k, confidence):
    sorted_inds = np.argsort(array)[-k:][::-1]
    confident_inds = [ind for ind in sorted_inds if array[ind] > confidence]
    return confident_inds


class Botmoji():
    def __init__(self, max_sentence_length=30):
        # Tokenizing using the dictionary
        with open(VOCAB_PATH, 'r') as f:
            self.vocabulary = json.load(f)

        self.st = SentenceTokenizer(self.vocabulary, max_sentence_length)

        # Loading the model
        self.model = torchmoji_emojis(PRETRAINED_PATH)

    def emojize_text(self, text, maxemojis, minconfidence):
        prob = self.encode(text)

        # Top emoji ID
        emoji_ids = top_emojis(prob, maxemojis, minconfidence)
        if len(emoji_ids) == 0:
            return ''

        # Map to emojis
        emojis = map(lambda x: EMOJIS[x], emoji_ids)
        return emoji.emojize(' '.join(emojis), use_aliases=True)

    def encode(self, text):
        # Running predictions
        tokenized, _, _ = self.st.tokenize_sentences([text])
        # Getting emojis probabilities
        prob = self.model(tokenized)[0]
        return prob

    def encode_multiple(self, texts):
        filtered_texts = ['_' if text == '' else text for text in texts]
        # Running predictions
        tokenized, _, _ = self.st.tokenize_sentences(filtered_texts)
        # Getting emojis probabilities
        prob = self.model(tokenized)
        return prob


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--text', type=str, required=True, help="Input text to emojize")
    argparser.add_argument('--maxlen', type=int, default=30, help="Max length of input text")
    argparser.add_argument('--maxemojis', type=int, default=5, help="Max number of top emojis to visualize")
    argparser.add_argument('--minconfidence', type=int, default=0.07, help="Minimum confidence in emoji")
    args = argparser.parse_args()

    botmoji = Botmoji(args.maxlen)
    probs = botmoji.encode(args.text)
    emojis = botmoji.emojize_text(args.text, args.maxemojis, args.minconfidence)
    print(probs)
    print(args.text, emojis)
