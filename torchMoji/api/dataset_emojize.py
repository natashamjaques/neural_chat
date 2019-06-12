# -*- coding: utf-8 -*-

""" Uses torchMoji to predict emojis for a dataset and save the softmax outputs to a file. """

from __future__ import print_function, division, unicode_literals
import json
import argparse

import numpy as np
import emoji
import pickle
import os

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


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filepath', type=str, required=False, help="Dataset file path")
    argparser.add_argument('--debuglen', type=int, default=5, help="Number of the sentences to show the output for")
    argparser.add_argument('--step', type=int, default=100, help="Number of steps for saving output")
    argparser.add_argument('--maxlen', type=int, default=30, help="Max length of input text")
    args = argparser.parse_args()
    sentence_probs = []
    retokenized_sentences = []
    output_path = os.path.join(os.path.dirname(args.filepath), 'sentence_emojis.pkl')
    retokenized_sentences_output_path = os.path.join(os.path.dirname(args.filepath), 'retokenized_sentences.pkl')

    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, args.maxlen)

    # Loading model
    model = torchmoji_emojis(PRETRAINED_PATH)

    sentences = pickle.load(open(args.filepath, 'rb'))
    idx = 0
    for conversation in sentences:
        idx += 1
        conversation_probs = []
        conversation_retokenized = []
        for sentence in conversation:
            try:
                tokenized, _, _ = st.tokenize_sentences([sentence])
            except:
                conversation_probs += [[1.0/len(EMOJIS)]*len(EMOJIS)]
                conversation_retokenized += [['']]
                continue
            prob = model(tokenized)[0]
            conversation_probs += [list(prob)]
            conversation_retokenized += [st.to_tokens(list(tokenized[0]))]
            if idx < args.debuglen:
                print(max(prob))
                print(prob)
                # Top emoji id
                emoji_ids = top_elements(prob, 1)
                # map to emojis
                emojis = map(lambda x: EMOJIS[x], emoji_ids)
                print('Retokenized: ', st.to_sentence(list(tokenized[0])))
                print(emoji.emojize("{} {}".format(sentence, ' '.join(emojis)), use_aliases=True))

        if idx % args.step == 0:
            print('Step: ', idx)
        sentence_probs += [conversation_probs]
        retokenized_sentences += [conversation_retokenized]
    print('Step: ', idx)
    pickle.dump(sentence_probs, open(output_path, 'wb'))
    pickle.dump(retokenized_sentences, open(retokenized_sentences_output_path, 'wb'))
