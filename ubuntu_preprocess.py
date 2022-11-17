# Load the Ubuntu dialog corpus
# Available from here:
# http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/ubuntu_dialogs.tgz

from multiprocessing import Pool
from pathlib import Path
from collections import OrderedDict
from urllib.request import urlretrieve
import os
import argparse
import tarfile
import pickle

from tqdm import tqdm
import pandas as pd

from model.utils import Tokenizer, Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, pad_sentences

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')
ubuntu_dir = datasets_dir.joinpath('ubuntu/')

ubuntu_meta_dir = ubuntu_dir.joinpath('meta/')
dialogs_dir = ubuntu_dir.joinpath('dialogs/')

# Tokenizer
tokenizer = Tokenizer('spacy')


def prepare_ubuntu_data():
    """Download and unpack dialogs"""

    tar_filename = 'ubuntu_dialogs.tgz'
    url = 'http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
    tarfile_path = ubuntu_dir.joinpath(tar_filename)
    metadata_url = 'https://raw.githubusercontent.com/rkadlec/ubuntu-ranking-dataset-creator/master/src/meta/'

    if not datasets_dir.exists():
        datasets_dir.mkdir()
    if not ubuntu_dir.exists():
        ubuntu_dir.mkdir()
    if not ubuntu_meta_dir.exists():
        ubuntu_meta_dir.mkdir()

    # Prepare Dialog data
    if not dialogs_dir.joinpath("10/1.tst").exists():
        # Download Dialog tarfile
        if not tarfile_path.exists():
            print(f"Downloading {url} to {tarfile_path}")
            urlretrieve(url, tarfile_path)
            print(f"Successfully downloaded {tarfile_path}")

        # Unpack tarfile
        if not dialogs_dir.exists():
            print("Unpacking dialogs ... (This can take 5~10 mins.)")
            with tarfile.open(tarfile_path) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=ubuntu_dir)
            print("Archive unpacked.")

    # Download metadata
    if not ubuntu_meta_dir.joinpath('trainfiles.csv').exists():
        print('Downloading metadata ... (This can take 5~10 mins.)')
        for filename in ['trainfiles.csv', 'valfiles.csv', 'testfiles.csv']:
            csv_path = ubuntu_meta_dir.joinpath(filename)
            print(f"Downloading {metadata_url+filename} to {csv_path}")
            urlretrieve(metadata_url + filename, csv_path)
            print(f"Successfully downloaded {csv_path}")

    print('Ubuntu Data prepared!')


def get_dialog_path_list(dataset='train'):
    if dataset == 'train':
        filename = 'trainfiles.csv'
    elif dataset == 'test':
        filename = 'testfiles.csv'
    elif dataset == 'valid':
        filename = 'valfiles.csv'
    with open(ubuntu_meta_dir.joinpath(filename)) as f:
        dialog_path_list = []
        for line in f:
            file, dir = line.strip().split(",")
            path = dialogs_dir.joinpath(dir, file)
            dialog_path_list.append(path)

    return dialog_path_list


def read_and_tokenize(dialog_path, min_turn=3):
    """
    Read conversation
    Args:
        dialog_path (str): path of dialog (tsv format)
    Return:
        dialogs: (list of list of str) [dialog_length, sentence_length]
        users: (list of str); [2]
    """
    with open(dialog_path, 'r', encoding='utf-8') as f:

        # Go through the dialog
        first_turn = True
        dialog = []
        users = []
        same_user_utterances = []  # list of sentences of current user
        dialog.append(same_user_utterances)

        for line in f:
            _time, speaker, _listener, sentence = line.split('\t')
            users.append(speaker)

            if first_turn:
                last_speaker = speaker
                first_turn = False

            # Speaker has changed
            if last_speaker != speaker:
                same_user_utterances = []
                dialog.append(same_user_utterances)

            same_user_utterances.append(sentence)
            last_speaker = speaker

        # All users in conversation (len: 2)
        users = list(OrderedDict.fromkeys(users))

        # 1. Concatenate consecutive sentences of single user
        # 2. Tokenize
        dialog = [tokenizer(" ".join(sentence)) for sentence in dialog]

        if len(dialog) < min_turn:
            print(f"Dialog {dialog_path} length ({len(dialog)}) < minimum required length {min_turn}")
            return []

    return dialog #, users


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--min_vocab_frequency', type=int, default=5)

    # Multiprocess
    parser.add_argument('--n_workers', type=int, default=os.cpu_count())

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers

    min_turn = 3

    # Download and unpack dialogs if necessary.
    prepare_ubuntu_data()

    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    for split_type in ['train', 'test', 'valid']:
        print(f'Processing {split_type} dataset...')
        split_data_dir = ubuntu_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)

        # List of dialogs (tsv)
        dialog_path_list = get_dialog_path_list(split_type)

        print(f'Tokenize.. (n_workers={n_workers})')
        def _tokenize_conversation(dialog_path):
            return read_and_tokenize(dialog_path)
        with Pool(n_workers) as pool:
            conversations = list(tqdm(pool.imap(_tokenize_conversation, dialog_path_list),
                                      total=len(dialog_path_list)))

        # Filter too short conversations
        conversations = list(filter(lambda x: len(x) >= min_turn, conversations))

        # conversations: padded_sentences
        # [n_conversations, conversation_length (various), max_sentence_length]

        # sentence_length: list of length of sentences
        # [n_conversations, conversation_length (various)]

        conversation_length = [min(len(conversation), max_conv_len)
                               for conversation in conversations]

        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))

        if split_type == 'train':
            print('Save Vocabulary...')
            vocab = Vocab(tokenizer)
            vocab.add_dataframe(conversations)
            vocab.update(max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(ubuntu_dir.joinpath('word2id.pkl'), ubuntu_dir.joinpath('id2word.pkl'))

        print('Done!')
