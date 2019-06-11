# -*- coding: utf-8 -*-

""" Use torchMoji to predict emojis from a single text input
"""

from __future__ import print_function, division, unicode_literals
import json
import argparse

import numpy as np
import emoji

from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_emojis
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

# Emoji map in emoji_overview.png
EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')

def top_emojis(array, k, confidence):
    sorted_inds = np.argsort(array)[-k:][::-1]
    confident_inds = [ind for ind in sorted_inds if array[ind] > confidence]
    return confident_inds

class Botmoji():
    def __init__(self, max_sentence_length=30):
        # Tokenizing using dictionary
        with open(VOCAB_PATH, 'r') as f:
            self.vocabulary = json.load(f)

        self.st = SentenceTokenizer(self.vocabulary, max_sentence_length)

        # Loading model
        self.model = torchmoji_emojis(PRETRAINED_PATH)

    def emojize_text(self, text, maxemojis, minconfidence):
        prob = self.encode(text)
        # Top emoji id
        emoji_ids = top_emojis(prob, maxemojis, minconfidence)
        if len(emoji_ids) == 0:
            return ''
        # map to emojis
        emojis = map(lambda x: EMOJIS[x], emoji_ids)
        return emoji.emojize(' '.join(emojis), use_aliases=True)

    def encode(self, text):
        # Running predictions
        tokenized, _, _ = self.st.tokenize_sentences([text])
        # Get emojis probability
        prob = self.model(tokenized)[0]
        return prob

    def encode_multiple(self, texts):
        filtered_texts = ['_' if text == '' else text for text in texts]
        # Running predictions
        tokenized, _, _ = self.st.tokenize_sentences(filtered_texts)
        # Get emojis probability
        prob = self.model(tokenized)
        return prob


# sample commandline run from the root folder:
# python torchMoji/examples/botmoji.py --text='Sample text. Does it encode it correctly?'
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
    print (probs)
    print(args.text, emojis)
