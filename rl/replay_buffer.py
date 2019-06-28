import numpy as np
import pandas as pd
import random
import re
import ast
from datetime import datetime

from model.utils import Vocab, Tokenizer, pad_tokens, detokenize

# String names of columns for s, a, r, s'
STUDY_COL = 'Study Key'
DATETIME_COL = 'Datetime'
CHAT_ID_COL = 'Chat ID'
BOT_RESPONSE_COL = 'Response'
USER_INPUT_COL = 'Message'
BAD_BOTS = ['bot1', 'bot2', 'bot3', 'bot4', 'pretzel']


class CsvReplayBuffer(object):
    def __init__(self, filepath, raw=True, history_len=10, study_key=None, 
                 config=None, max_sentence_length=30, rewards=None, 
                 reward_weights=None, model_averaging=False):
        """
        Args:
            filepath: location of the .csv to load experience from
            raw: if True, using raw data from website. Will applying filtering
                and cleaning functions
            frame_history_len: int
                Number of memories to be retried for each observation.
        """
        self.filepath = filepath
        self.raw = raw
        self.study_key = study_key
        self.config = config
        self.history_len = history_len
        self.max_sentence_length = max_sentence_length
        self.rewards = rewards
        self.reward_weights = reward_weights
        self.model_averaging = model_averaging

        print("Loading experience from file", filepath)
        self.buffer = pd.read_csv(filepath)

        if study_key is not None:
            print("Limiting to study", study_key)
            self.buffer = self.buffer[self.buffer[STUDY_COL] == study_key]

        if raw:
            self.buffer = process_raw_csv(self.buffer, self.history_len)

        # Will be used for training RL model
        if config:
            # Load vocab
            self.vocab = Vocab()
            self.vocab.load(config.word2id_path, config.id2word_path)
            self.config.vocab_size = self.vocab.vocab_size

            self.tokenizer = Tokenizer('spacy')

            self.preprocess_model_inputs()

        self.ordered_idx = 0
        self.ordered_passes_through_data = 0

    def sample(self, batch_size):
        sample_idx = np.random.choice(np.arange(len(self.state)), 
                                      size=batch_size)
        batch = {'state': self.state[sample_idx], 
                 'state_lens': self.state_lens[sample_idx], 
                 'action': self.action[sample_idx],
                 'action_lens': self.action_lens[sample_idx],
                 'rewards': self.reward[sample_idx], 
                 'next_state': self.next_state[sample_idx],
                 'next_state_lens': self.next_state_lens[sample_idx],
                 'done': self.dones[sample_idx]}

        if self.model_averaging:
            batch['model_averaged_probs'] = \
                [self.model_averaged_probs[i] for i in sample_idx]

        return batch

    def get_batch_in_order(self, batch_size):
        if self.ordered_idx > len(self.buffer):
            self.ordered_passes_through_data += 1
            print('Just finished a complete ordered pass through the data.')
            print('Have now made', self.ordered_passes_through_data, 'passes.')
            self.ordered_idx = 0 

        sample_idx = np.arange(
            self.ordered_idx,
            min(self.ordered_idx + batch_size, len(self.state)))

        batch = {'state': self.state[sample_idx], 
                 'state_lens': self.state_lens[sample_idx], 
                 'action': self.action[sample_idx],
                 'action_lens': self.action_lens[sample_idx],
                 'rewards': self.reward[sample_idx], 
                 'next_state': self.next_state[sample_idx],
                 'next_state_lens': self.next_state_lens[sample_idx],
                 'done': self.dones[sample_idx]}

        self.ordered_idx += batch_size

        return batch


    def preprocess_model_inputs(self):
        # Fill NAN actions with empty string
        self.buffer['action'] = self.buffer['action'].fillna("")
        # Cannot learn from examples where no action was taken
        self.buffer = self.buffer[self.buffer['action'] != '']

        # Convert states
        self.state, self.state_lens = self.convert_list_to_model_input(
            self.buffer['state'].tolist())
        (self.next_state, 
         self.next_state_lens) = self.convert_list_to_model_input(
            self.buffer['next_state'].tolist())
        
        # Convert actions
        self.action, self.action_lens = self.convert_list_to_model_input(
            self.buffer['action'].tolist(), list_of_lists=False)

        self.reward = np.zeros(len(self.buffer))
        if self.rewards is not None:
            for i, r_name in enumerate(self.rewards):
                r_weight = float(self.reward_weights[i])
                r_val = np.array(self.buffer[r_name].tolist())
                self.reward += r_val * r_weight

        self.dones = np.array(self.buffer['done'])

        if self.model_averaging:
            self.model_averaged_probs = self.extract_model_averaged_probs()
    
    def convert_list_to_model_input(self, sentences_list, list_of_lists=True):
        try:
            if list_of_lists:
                if not self.raw:
                    sentences_list = [ast.literal_eval(x) for x in sentences_list]

                # Add periods if they are missing
                # Not doing this so model can learn to be robust to bad human/model inputs
                # for i in range(len(sentences_list)):
                #     sentences_list[i] = [s.rstrip() for s in sentences_list[i]] # Remove trailing whitespace
                #     for j, s in enumerate(sentences_list[i]):
                #         if s[-1] not in ['.', '!', '?']:
                #             sentences_list[i][j] = s + '.'

                tokenized_list = [[self.tokenizer(sent) for sent in sentences] for sentences in sentences_list]
                trimmed_list = [[tokens[:self.max_sentence_length-1] for tokens in tokenized] for tokenized in tokenized_list]
                lens = [[len(t)+1 for t in trimmed] for trimmed in trimmed_list]
                padded_list = [[pad_tokens(tokens) for tokens in trimmed] for trimmed in trimmed_list]
                coded = [self.sent2id(padded) for padded in padded_list]
            else:
                tokens = [self.tokenizer(sent) for sent in sentences_list]
                trimmed = [t[:self.max_sentence_length-1] for t in tokens]
                lens = [len(t)+1 for t in trimmed]
                padded = [pad_tokens(t) for t in trimmed]
                coded = self.sent2id(padded)

            coded = np.array([np.array(c) for c in coded])
            lens = np.array([np.array(l) for l in lens])

            return coded, lens
        except Exception as e:
            print("error in convert_list_to_model_input")
            print(str(e))
            import pdb; pdb.set_trace()

    def sent2id(self, sentences):
        """word => word id"""
        return [self.vocab.sent2id(sentence) for sentence in sentences]
    
    def id2sent(self, tokens):
        """word => word id"""
        return [self.vocab.decode(sent) for sent in tokens]

    def extract_model_averaged_probs(self):
        if self.config.separate_datasets:
            probs_col = 'model_averaged_probs_' + self.config.data
        else:
            probs_col = 'model_averaged_probs'
        
        assert (probs_col in self.buffer.columns.values), "Error!" + \
            "This csv does not contain model_averaged_probs."

        # Must convert strings to numbers
        model_averaged_probs = []
        for string in self.buffer[probs_col].tolist():
            for ch in ['\n', '[', ']']:
                string = string.replace(ch, '')
            string_probs = string.split(' ')
            string_probs = [s for s in string_probs if s != '']
            try:
                model_averaged_probs.append([float(x) for x in string_probs])
            except Exception as e:
                print(e)
                print('Could not convert all strings to floats:')
                print(string_probs)
                import pdb; pdb.set_trace()
        
        return model_averaged_probs

def process_raw_csv(df, max_convo_len=10):
    # Convert datetimes
    df[DATETIME_COL] = [pd.to_datetime(d) for d in df[DATETIME_COL]]
    df[BOT_RESPONSE_COL] = df[BOT_RESPONSE_COL].fillna("")

    # Remove bad bots
    for bot in BAD_BOTS:
        df = df[df['Chatbot ID'] != bot]

    conversations = df[CHAT_ID_COL].unique()

    processed_df = None

    # Extract conversations
    for conv_id in conversations:
        conv = df[df[CHAT_ID_COL] == conv_id]
        conv = conv.sort_values(DATETIME_COL)

        # Strip emojis from bot responses
        responses = [strip_emojis(a) for a in conv[BOT_RESPONSE_COL]]
        conv['action'] = responses

        # Make whole convo
        context = []
        for i,o in zip(conv[USER_INPUT_COL], responses):
            context.extend([i,o])
        
        # Append a blank for last user response to keep indexing consistent
        context.append("")

        # Find RL state / next state pairs
        states = []
        next_states = []
        for i in range(len(conv)):
            states.append(context[:1+i*2][-(max_convo_len-2):])
            next_states.append(context[:3+i*2][-max_convo_len:])
        conv['state'] = states
        conv['next_state'] = next_states

        dones = [0] * len(states)
        dones[len(states)-1] = 1
        conv['done'] = dones

        # Attach to big df
        if processed_df is None:
            processed_df = conv
        else:
            processed_df = pd.concat([processed_df, conv])

    return processed_df

def strip_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
       "]+", flags=re.UNICODE)
    stripped = emoji_pattern.sub(r'', text) # no emoji
    return stripped.lstrip()


