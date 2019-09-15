"""Script which converts reddit data into ParlAI format. datapath should point
to neural_chat/datasets/reddit_casual and the ParlAI format data will be saved
to neural_chat/ParlAI/data/redditcasual
"""
import os
import pickle
from pathlib import Path
from random import sample

import parlai.core.build_data as build_data


def add_interaction(conv):
    global processed
    conv_len = len(conv) if len(conv) % 2 == 0 else len(conv)-1
    if not conv_len: return
    _conv = conv[:conv_len]

    pairs = list(zip(_conv[::2], _conv[1::2]))
    for p in pairs:
        line = f'text:{p[0]}\tlabels:{p[1]}\n'
        processed += line
    # undo new line
    processed = processed.strip('\n')
    processed += '\tepisode_done:True\n'


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'redditcasual')
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        project_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        data_dir = project_dir.joinpath('datasets/reddit_casual/')

        for split in ['valid', 'test', 'train']:
            path = data_dir.joinpath(split, 'raw_sentences.pkl')
            data = pickle.load(open(path, 'rb'))

            if split in {'valid', 'test'}:
                data = sample(data, 2000)

            save_path = project_dir.joinpath('ParlAI', 'data', 'redditcasual', f'{split}.txt')
            with open(save_path, 'w', encoding='utf-8') as f:
                for conv in data:
                    processed = ''

                    # clean
                    conv = [' '.join(s.split()).lower() for s in conv]

                    # Create one perspective.
                    # For example, the conversation x1, y1, x2, y2, x3 becomes:
                    # x1 y1
                    # x2 y2
                    add_interaction(conv)

                    # Create other perspective.
                    # For example, the conversation x1, y1, x2, y2, x3 becomes:
                    # y1 x2
                    # y2 x3
                    add_interaction(conv[1:])

                    f.write(processed)

    # mark the data as built
    build_data.mark_done(dpath, version_string=version)
