""" This file loads all available chatbots and computes the aggregated 
probability they place on each sentence in the dataset, weighted by each model's
predefined score or proportion in the data. """

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

from model.models import VariationalModels
from model.utils import to_var, EOS_ID

import replay_buffer
import run_rl

TEST_WEIGHTS = {
    'emotion_VHRED_cornell': .5, 
    'infersent_HRED_cornell': .1,
    'emoinfer_VHCR_casual': .2, 
    'emoinfer_VHRED_casual': .3,
}

# From actual study, weights are quality ratings
PREDEFINED_MODEL_WEIGHTS = {
    # Reddit
    'emotion_HRED_casual': 3.125,
    'infersent_VHRED_casual': 3.047619048,
    'emotion_VHRED_casual': 2.913043478,
    # 'emoinfer_VHRED_casual': 2.863636364,
    # 'emoinfer_HRED_casual': 2.714285714,
    # 'VHRED_reddit_casual': 2.694444444,
    # 'emoinfer_VHCR_casual': 2.692307692,
    # 'VHCR_reddit_casual': 2.615384615,
    # 'infersent_VHCR_casual': 2.533333333,
    # 'HRED_reddit_casual': 2.527272727,
    # 'emotion_VHCR_casual': 2.5,
    # 'infersent_HRED_casual': 2.5,

    # Cornell
    'emoinfer_VHCR_cornell': 2.547619048,
    'infersent_VHCR_cornell': 2.541666667,
    'emotion_VHRED_cornell': 2.529411765,
    # 'emotion_VHCR_cornell': 2.5,
    # 'infersent_HRED_cornell': 2.388888889,
    # 'infersent_VHRED_cornell': 2.380952381,
#     'emoinfer_HRED_cornell': 2.346938776,
#     'emoinfer_VHRED_cornell': 2.333333333,
#     'emotion_HRED_cornell': 2.238095238,
#     'HRED_cornell': 2.181818182,
#     'VHCR_cornell': 2.132075472,
#     'VHRED_cornell': 2.02173913,
}

# Below weights are based on pilot data, before running the study
# PREDEFINED_MODEL_WEIGHTS = {
#     # Cornell
#     'HRED_cornell': 211,
#     'VHRED_cornell': 202,
#     'VHCR_cornell': 159,
#     'emotion_HRED_cornell': 157,
#     'emotion_VHRED_cornell': 998,  # 1061 (ELSA_) + 137 (emotion_)
#     'emotion_VHRED_cornell_hparam': 200, # allocated from above
#     'emotion_VHCR_cornell': 173,
#     'infersent_HRED_cornell': 48,  # No way to distinguish from casual so gave 2/3 of total of 73 here
#     'infersent_VHCR_cornell': 145,  # No way to distinguish from casual so gave 2/3 of total of 220 here
#     'infersent_VHRED_cornell': 195,  # No way to distinguish from casual so gave 2/3 of total of 296 here
#     'emoinfer_HRED_cornell': 36,
#     'emoinfer_VHRED_cornell': 261,  # No way to distinguish from casual so gave 2/3 of total here
#     'emoinfer_VHCR_cornell': 283,
#     'input_only_HRED_cornell': 54,  # No way to distinguish so gave 1/2 of total of 108 here
#     'input_only_VHRED_cornell': 28,  # No way to distinguish so gave 1/2 of total of 55 here
#     'input_only_VHCR_cornell': 35,  # No way to distinguish so gave 1/2 of total of 69 here

#     # Reddit casual
#     'HRED_reddit_casual': 113,
#     'VHRED_reddit_casual': 80,
#     'VHCR_reddit_casual': 67,
#     'emotion_HRED_casual': 59,
#     'emotion_VHRED_casual': 66,  # 43 + 23 (ELSA_VHRED_reddi)
#     'emotion_VHCR_casual': 155,  # 143 + 12 (ELSA_VHCR_reddit)
#     'infersent_HRED_casual': 25,  # No way to distinguish from cornell so gave 1/3 of total of 73 here
#     'infersent_VHCR_casual': 75,  # No way to distinguish from cornell so gave 1/3 of total of 220 here
#     'infersent_VHRED_casual': 101,  # No way to distinguish from cornell so gave 1/3 of total of 296 here
#     'emoinfer_HRED_casual': 114,
#     'emoinfer_VHRED_casual': 112,
#     'emoinfer_VHCR_casual': 29,
#     'input_only_HRED_casual': 54,  # No way to distinguish so gave 1/2 of total of 108 here
#     'input_only_VHRED_casual': 28,  # No way to distinguish so gave 1/2 of total of 55 here
#     'input_only_VHCR_casual': 35,  # No way to distinguish so gave 1/2 of total of 69 here
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experience_path', type=str, default=None,
                        help='Path to a .csv containing experiences')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save csv with computed reward')
    parser.add_argument('--models_base_path', type=str, default=None,
                        help='Base path to checkpoints for model averaging.')
    parser.add_argument('--model_weights', type=str, default='predefined',
                        help="How to weight the probabilites of each model. Can"
                             "be 'predefined' or 'proportional'")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--separate_datasets', action='store_true',
                        help="If true, don't merge probabilities of models " + \
                             " from different datasets")
    parser.set_defaults(separate_datasets=False)
    return parser.parse_args()


def get_default_rl_config():
    return run_rl.parse_config_args()


class ModelAverager:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.separate_datasets = kwargs.separate_datasets
        
        # Load chatbots
        os.environ['BASE_PATH'] = kwargs.models_base_path
        from model.model_avg_chatbots import chatbots
        self.chatbots = chatbots

        # Load dataset buffers
        self.dataset_buffs = {}
        for name, model in chatbots.items():
            if model.config.data not in self.dataset_buffs:
                print('\nCreating buffer for dataset', model.config.data)
                buffer = replay_buffer.CsvReplayBuffer(
                    kwargs.experience_path, raw=False, config=model.config)
                print("Buffer contains", len(buffer.buffer), "responses.")
                self.dataset_buffs[model.config.data] = buffer
                self.dataset_len = len(buffer.buffer)

        # Get weights
        if kwargs.model_weights == 'proportional':
            # Compute proportion of each bot in the dataset
            self.weights = compute_bot_proportion(chatbots, csv_buff.buffer)
        elif kwargs.model_weights == 'predefined':
            self.weights = PREDEFINED_MODEL_WEIGHTS 

            # Discard bots that are not going to be aggregated based on weights
            to_delete = [c for c in self.chatbots.keys() if c not in self.weights]
            for c in to_delete:
                del self.chatbots[c]
        else:
            print('Error!', kwargs.model_weights, 'is not a valid way to weight',
                "models. Please use 'proportional', or 'predefined'.")

        # Ensure weights sum to 1 to maintain a valid probability distribution
        weight_values = [self.weights[k] for k in self.weights.keys()]
        norm_weights = weight_values / np.sum(weight_values)
        for i, k in enumerate(self.weights.keys()):
            self.weights[k] = norm_weights[i]

    def compute_bot_proportion(self):
        pass

    def average_models(self):
        if self.separate_datasets:
            averaged_probs = dict.fromkeys(self.dataset_buffs)
            # Cannot assign default value of [] because the lists will be shared
            for k in self.dataset_buffs.keys():
                averaged_probs[k] = []
        else:
            averaged_probs = []
        
        i = 0
        while i < self.dataset_len:
            # Get different versions of the batch for the different tokenization
            # schemes for each dataset
            dataset_batches = {}
            for dataset, buff in self.dataset_buffs.items():
                batch = buff.get_batch_in_order(
                    self.kwargs.batch_size)
                dataset_batches[dataset] = self.extract_sentence_data(batch)
                action_lens = dataset_batches[dataset][1]

            if self.separate_datasets:
                batch_probs = dict.fromkeys(self.dataset_buffs)
            else:
                batch_probs = None

            for name, bot in self.chatbots.items():
                model_probs = self.run_model_on_sentences(
                    bot, dataset_batches[bot.config.data])  # [total words]

                weighted_probs = self.weights[name] * model_probs.cpu().numpy()

                if self.separate_datasets:
                    if batch_probs[bot.config.data] is None:
                        batch_probs[bot.config.data] = weighted_probs
                    else:
                        batch_probs[bot.config.data] += weighted_probs
                else:
                    if batch_probs is None:
                        batch_probs = weighted_probs
                    else:
                        batch_probs += weighted_probs

            # Put everything back into the appropriate sentence
            # [batch_size, sentence length]
            start = np.cumsum([0, *action_lens[:-1]])
            if self.separate_datasets:
                for k in averaged_probs.keys():
                    sentence_probs = [batch_probs[k][s:s+l] 
                              for s, l in zip(start, action_lens)]

                    # Accumulate this batch
                    averaged_probs[k].extend(sentence_probs)
            else:
                sentence_probs = [batch_probs[s:s+l] 
                              for s, l in zip(start, action_lens)]
                # Accumulate this batch
                averaged_probs.extend(sentence_probs)

            i += self.kwargs.batch_size

        # Double check we got the appropriate probabilities matching the 
        # sentence lengths for everything in the buffer
        for d, b in self.dataset_buffs.items():
            print("Double checking", d, "dataset buffer")
            for i in range(len(b.action)):
                action_len = list(b.action[i]).index(EOS_ID) + 1
                if self.separate_datasets:
                    avg_len = len(averaged_probs[d][i])
                else:
                    avg_len = len(averaged_probs[i])
                assert (action_len == avg_len), error_message
                error_message = "There are " + str(action_len) + \
                    " words in the buffer at index " + str(i) + " but " + \
                    str(avg_len) + " probs"
        print("Model-averaged action probabilities computed!")

        for dataset, buff in self.dataset_buffs.items():
            if not self.separate_datasets:
                buff.buffer['model_averaged_probs'] = averaged_probs
            else:
                for d, avg_p in averaged_probs.items():
                    buff.buffer['model_averaged_probs_' + d] = avg_p
            return buff  # Can return any buffer, relevant info is the same

    def extract_sentence_data(self, batch):
        with torch.no_grad():
            # Extract batch info 
            actions = to_var(torch.LongTensor(batch['action']))  # [batch_size]
            action_lens = batch['action_lens']

            conversations = [np.concatenate(
                (conv, np.atleast_2d(batch['action'][i]))) 
                for i, conv in enumerate(batch['state'])]
            sent_lens = [np.concatenate(
                (lens, np.atleast_1d(batch['action_lens'][i]))) 
                for i, lens in enumerate(batch['state_lens'])]
            target_conversations = [conv[1:] for conv in conversations]
            targets = [sent for conv in target_conversations for sent in conv]
            targets = to_var(torch.LongTensor(targets))
            conv_lens = [len(c) - 1 for c in conversations]
            conv_lens = to_var(torch.LongTensor(conv_lens))

            # Compute non-variational inputs
            hred_convs = [conv[:-1] for conv in conversations]
            hred_sent_lens = np.concatenate([l[:-1] for l in sent_lens])
            hred_sent_lens = to_var(torch.LongTensor(hred_sent_lens))
            hred_sentences = [sent for conv in hred_convs for sent in conv]
            hred_sentences = to_var(torch.LongTensor(hred_sentences))
            
            # Compute variational inputs
            sent_lens = np.concatenate([l for l in sent_lens])
            sent_lens = to_var(torch.LongTensor(sent_lens))
            sentences = [sent for conv in conversations for sent in conv]
            sentences = to_var(torch.LongTensor(sentences))

            return (actions, action_lens, sentences, sent_lens, hred_sentences, 
                    hred_sent_lens, targets, conv_lens)

    def run_model_on_sentences(self, bot, batch_tensors):
        with torch.no_grad():
            (actions, action_lens, sentences, sent_lens, hred_sentences, 
            hred_sent_lens, targets, conv_lens) = batch_tensors
            if bot.config.model not in VariationalModels:
                sentences = hred_sentences
                sent_lens = hred_sent_lens

            # Run model
            outputs = bot.solver.model(sentences, sent_lens, conv_lens, targets, 
                                    rl_mode=True)
            logits = outputs[0]

            # Index to get only output values for actions taken (last sentence
            # in each conversation)
            start = torch.cumsum(torch.cat(
                (to_var(conv_lens.data.new(1).zero_()), conv_lens[:-1])), 0)
            action_logits = torch.stack(
                [logits[s+l-1, :, :]
                for s, l in zip(start.data.tolist(), conv_lens.data.tolist())], 
                0)  # [num_sentences, max_sent_len, vocab_size]

            # Limit by actual sentence length (remove padding) and flatten into
            # long list of words
            word_logits = torch.cat(
                [action_logits[i, :l, :] for i, l in enumerate(action_lens)], 
                0)  # [total words, vocab_size]
            word_actions = torch.cat(
                [actions[i, :l] for i, l in enumerate(action_lens)], 
                0)  # [total words]

            # Take softmax to get probability distribution 
            # [total_words, vocab_size]
            word_probs = torch.nn.functional.softmax(word_logits, 1)

            # Extract q values corresponding to actions taken
            relevant_words = word_probs.gather(
                1, word_actions.unsqueeze(1)).squeeze()  # [total words]

            return relevant_words


if __name__ == '__main__':
    """Load experience replay buffer from file and compute rewards on it"""
    kwargs = parse_args()

    ma = ModelAverager(kwargs)

    # Compute the model averaged probabilities
    buffer = ma.average_models()

    # Save to processed csv file
    buffer.buffer.to_csv(kwargs.save_path)