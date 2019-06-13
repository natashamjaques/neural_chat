# Preprocess cornell movie dialogs dataset

from multiprocessing import Pool
import argparse
import pickle
import random
import os
import json
from urllib.request import urlretrieve
from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm
from model.utils import Tokenizer, Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, pad_sentences
from subprocess import call

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')
cornell_dir = datasets_dir.joinpath('cornell/')

# Tokenizer
tokenizer = Tokenizer('spacy')


def shortcut_download(dataset):
    """Download and unpack pre-processed dataset"""

    zip_url = f'https://affect.media.mit.edu/neural_chat/datasets/{dataset}.zip'
    zipfile_dir = os.path.join(datasets_dir, dataset)
    zipfile_path = os.path.join(datasets_dir, f'{dataset}.zip')

    if not datasets_dir.exists():
        datasets_dir.mkdir()

    # Prepare Dialog data
    if not zipfile_dir.exists():
        print(f'Downloading {zip_url} to {zipfile_path}')
        urlretrieve(zip_url, zipfile_path)
        print(f'Successfully downloaded {zipfile_path}')

        zip_ref = ZipFile(zipfile_path, 'r')
        zip_ref.extractall(datasets_dir)
        zip_ref.close()

    else:
        print('Directory already exists. Aborting download.')


def prepare_cornell_data():
    """Download and unpack dialogs"""

    zip_url = 'http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip'
    zipfile_path = datasets_dir.joinpath('cornell.zip')

    if not datasets_dir.exists():
        datasets_dir.mkdir()

    # Prepare Dialog data
    if not cornell_dir.exists():
        print(f'Downloading {zip_url} to {zipfile_path}')
        urlretrieve(zip_url, zipfile_path)
        print(f'Successfully downloaded {zipfile_path}')

        zip_ref = ZipFile(zipfile_path, 'r')
        zip_ref.extractall(datasets_dir)
        zip_ref.close()

        datasets_dir.joinpath('cornell movie-dialogs corpus').rename(cornell_dir)

    else:
        print('Cornell Data prepared!')


def load_lines(file_name,
               fields=["lineID", "characterID", "movieID", "character", "text"],
               delimiter=" +++$+++ "):
    """
    Args:
        file_name (str): file to load
        fields (set<str>): fields to extract
        delimiter (str): delimiter string
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    lines = {}

    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(delimiter)

            # Extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]

            lines[line_obj['lineID']] = line_obj

    return lines


def load_conversations(file_name, lines,
                       fields=["character1ID", "character2ID", "movieID", "utteranceIDs"],
                       delimiter=" +++$+++ "):
    """
    Args:
        file_name (str): file to load
        fields (set<str>): fields to extract
        delimiter (str): delimiter string
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    conversations = []

    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(delimiter)

            # Extract fields
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]

            # Convert string to list (conv_obj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            line_ids = eval(conv_obj["utteranceIDs"])

            # Reassemble lines
            conv_obj["lines"] = []
            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])

            conversations.append(conv_obj)

    return conversations


def train_valid_test_split_by_conversation(conversations, split_ratio=[0.8, 0.1, 0.1]):
    """Train/Validation/Test split by randomly selected movies"""

    train_ratio, valid_ratio, test_ratio = split_ratio
    assert train_ratio + valid_ratio + test_ratio == 1.0

    n_conversations = len(conversations)

    # Random shuffle movie list
    random.seed(0)
    random.shuffle(conversations)

    # Train / Validation / Test Split
    train_split = int(n_conversations * train_ratio)
    valid_split = int(n_conversations * (train_ratio + valid_ratio))

    train = conversations[:train_split]
    valid = conversations[train_split:valid_split]
    test = conversations[valid_split:]

    print(f'Train set: {len(train)} conversations')
    print(f'Validation set: {len(valid)} conversations')
    print(f'Test set: {len(test)} conversations')

    return train, valid, test


def tokenize_conversation(lines):
    sentence_list = [tokenizer(line['text']) for line in lines]
    return sentence_list


def pad_sentences(conversations, max_sentence_length=30, max_conversation_length=10):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        sentence_length = [min(len(sentence) + 1, max_sentence_length)  # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


def load_conversations_cornell(cornell_dir):
    # Download and extract dialogs if necessary.
    prepare_cornell_data()

    print("Loading lines")
    lines = load_lines(cornell_dir.joinpath("movie_lines.txt"))
    print('Number of lines:', len(lines))

    print("Loading conversations...")
    conversations = load_conversations(cornell_dir.joinpath("movie_conversations.txt"), lines)
    print('Number of conversations:', len(conversations))
    return conversations


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Split Ratio
    split_ratio = [0.8, 0.1, 0.1]

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--min_vocab_frequency', type=int, default=5)

    # Multiprocess
    parser.add_argument('--n_workers', type=int, default=os.cpu_count())

    # Input dataset
    parser.add_argument('--dataset', type=str, default='cornell')

    # Bypassing pre-processing by directly downloading all the files
    parser.add_argument('--shortcut', action="store_true", default=False,
                        help="Whether to download the preprocessed dataset instead.")

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers

    if args.shortcut:
        shortcut_download(args.dataset)
    else:
        conversations = []
        # Cornell data
        if args.dataset == 'cornell':
            dataset_dir = cornell_dir
            conversations = load_conversations_cornell(cornell_dir)
        # Reddit_casual data
        else:
            dataset_dir = datasets_dir.joinpath(args.dataset)
            with open('datasets/{}/{}.json'.format(args.dataset, args.dataset), 'r') as f:
                conversations = json.load(f)

        print('Train/Valid/Test Split')
        train, valid, test = train_valid_test_split_by_conversation(conversations, split_ratio)

        def to_pickle(obj, path):
            with open(path, 'wb') as f:
                pickle.dump(obj, f)

        for split_type, conv_objects in [('train', train), ('valid', valid), ('test', test)]:
            print(f'Processing {split_type} dataset...')
            split_data_dir = dataset_dir.joinpath(split_type)
            split_data_dir.mkdir(exist_ok=True)

            print(f'Tokenize.. (n_workers={n_workers})')

            def _tokenize_conversation(conv):
                return tokenize_conversation(conv['lines'])

            with Pool(n_workers) as pool:
                conversations = list(tqdm(pool.imap(_tokenize_conversation, conv_objects),
                                         total=len(conv_objects)))

            conversation_length = [min(len(conv['lines']), max_conv_len)
                                   for conv in conv_objects]

            raw_sentences = [[line['text'] for line in conv['lines'][0:min(len(conv['lines']), max_conv_len)]]
                             for conv in conv_objects]

            sentences, sentence_length = pad_sentences(
                conversations,
                max_sentence_length=max_sent_len,
                max_conversation_length=max_conv_len)

            print('Saving preprocessed data at', split_data_dir)
            to_pickle(raw_sentences, split_data_dir.joinpath('raw_sentences.pkl'))
            to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
            to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
            to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))

            if split_type == 'train':

                print('Save Vocabulary...')
                vocab = Vocab(tokenizer)
                vocab.add_dataframe(conversations)
                vocab.update(max_size=max_vocab_size, min_freq=min_freq)

                print('Vocabulary size: ', len(vocab))
                vocab.pickle(dataset_dir.joinpath('word2id.pkl'), dataset_dir.joinpath('id2word.pkl'))

        print('Done downloading and pre-processing dataset.')

        print('Inferring TorchMoji encoding for dataset...')
        torchmoji_export_script = os.path.join(os.path.join('torchMoji', 'api'), 'dataset_emojize.py')
        for split_type in ['train', 'valid', 'test']:
            filepath = os.path.join(os.path.join(dataset_dir, split_type), 'raw_sentences.pkl')
            call(["python", torchmoji_export_script, f'--filepath={filepath}'])
        print('Done exporting TorchMoji embedding.')

        print('Inferring InferSent encoding for dataset...')
        infersent_export_script = os.path.join(os.path.join('inferSent', 'api'), 'export_dataset_embeddings.py')
        for split_type in ['train', 'valid', 'test']:
            filepath = os.path.join(os.path.join(dataset_dir, split_type), 'raw_sentences.pkl')
            call(["python", infersent_export_script, f'--filepath={filepath}'])
        infersent_reduction_script = os.path.join(os.path.join('inferSent', 'api'), 'reduce_embeddings_dimension.py')
        call(["python", infersent_reduction_script, f'--dataset={dataset_dir} --savepca --exportembeddings'])
        print('Done exporting InferSent embedding.')

        print('Successfully completed!')
