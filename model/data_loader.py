import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from model.utils import PAD_ID, UNK_ID, SOS_ID, EOS_ID
import numpy as np

empty_sentence = ['<eos>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
                  '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
                  '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
default_max_convo_length = 10


class DialogDataset(Dataset):
    def __init__(self, sentences, conversation_length, sentence_length, vocab,
                 data=None, emojis=None, infersent=None):

        # [total_data_size, max_conversation_length, max_sentence_length]
        # tokenized raw text of sentences
        self.sentences = sentences
        self.vocab = vocab

        # conversation length of each batch
        # [total_data_size]
        self.conversation_length = conversation_length

        # list of length of sentences
        # [total_data_size, max_conversation_length]
        self.sentence_length = sentence_length
        self.data = data

        # Emoji vector for each sentence
        self.emojis = emojis

        # Infersent embedding vector for each sentence
        self.infersent = infersent

        # There is no emoji annotation for blank sentences,
        # so need to remove these
        if emojis is not None or infersent is not None:
            if emojis is not None: assert len(self.emojis) == len(self.sentences)
            if infersent is not None: assert len(self.infersent) == len(self.sentences)

            if self.data is not None:
                print('Warning! Unpredictable behavior when input needs filtering and data variable present')

            to_delete = []
            for i in range(len(self.sentences)):
                if ((emojis and len(self.emojis[i]) != len(self.sentences[i])) or
                        (infersent and len(self.infersent[i]) != len(self.sentences[i]))):
                    non_blanks = [s for s in self.sentences[i] if s != empty_sentence]
                    if len(non_blanks) < 2:
                        # Need to totally discard this row
                        to_delete.append(i)
                    else:
                        keep_indices = [j for j, s in enumerate(sentences[i]) if s != empty_sentence]
                        self.sentence_length[i] = [self.sentence_length[i][x] for x in keep_indices]
                        self.conversation_length[i] = len(non_blanks)
                        self.sentences[i] = non_blanks

                        if emojis is not None and len(emojis[i]) != len(non_blanks):
                            if len(emojis[i]) > default_max_convo_length:
                                self.emojis[i] = self.emojis[i][:len(non_blanks)]
                            else:
                                print('Number of emoji sentences', len(emojis[i]),
                                      'differs from length of sentences', len(non_blanks),
                                      'even with blank sentences removed and emojis trimmed.')

                        if infersent is not None and len(infersent[i]) != len(non_blanks):
                            self.infersent[i] = [self.infersent[i][x] for x in keep_indices]

                            if len(self.infersent[i]) != len(non_blanks):
                                print('Number of infersent sentences', len(infersent[i]),
                                      'differs from length of sentences', len(non_blanks),
                                      'even with blank sentences removed and infersent trimmed.')

            # Delete the necessary indices
            print('Removing', len(to_delete), "conversations with too many blank lines from the dataset.")
            for idx in to_delete:
                del self.sentences[idx]
                del self.emojis[idx]
                del self.conversation_length[idx]
                del self.sentence_length[idx]
                if infersent is not None:
                    del self.infersent[idx]

        self.len = len(self.sentences)
        assert self.len == len(self.conversation_length) == len(self.sentence_length)

        if emojis is not None:
            assert self.len == len(self.emojis)
        if infersent is not None:
            assert self.len == len(self.infersent)

    def __getitem__(self, index):
        """Return Single data sentence"""
        # [max_conversation_length, max_sentence_length]
        sentence = self.sentences[index]
        conversation_length = self.conversation_length[index]
        sentence_length = self.sentence_length[index]

        # word => word_ids
        sentence = self.sent2id(sentence)

        emojis = None
        infersent = None
        if self.emojis:
            emojis = self.emojis[index]
        if self.infersent:
            infersent = self.infersent[index]

        return sentence, conversation_length, sentence_length, emojis, infersent

    def __len__(self):
        return self.len

    def sent2id(self, sentences):
        """word => word id"""
        # [max_conversation_length, max_sentence_length]
        return [self.vocab.sent2id(sentence) for sentence in sentences]


def get_loader(sentences, conversation_length, sentence_length, vocab, batch_size=100, data=None,
               shuffle=True, emojis=None, infersent=None):
    """Load DataLoader of given DialogDataset"""

    def collate_fn(data):
        """
        Collate list of data in to batch
        Args:
            data: list of tuple(source, target, conversation_length, source_length, target_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)

        # Separate
        sentences, conversation_length, sentence_length, emojis, infersent = zip(*data)

        # return sentences, conversation_length, sentence_length.tolist()
        return sentences, conversation_length, sentence_length, emojis, infersent

    dataset = DialogDataset(sentences, conversation_length,
                            sentence_length, vocab, data=data,
                            emojis=emojis, infersent=infersent)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader