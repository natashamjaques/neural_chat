import sys
import os
import random
import pickle
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model.solver import VariationalSolver
from model.utils import to_var, EOS_ID, TensorboardWriter, convert_old_checkpoint_format, Vocab
from model.data_loader import get_loader
import hrl_rewards
from hrl_rewards import normalizeZ, discount
from Toxicity.toxic import NBLogisticRegression, NBTfidfVectorizer, tokenize


class REINFORCETuner:
    def __init__(self, config, val_config):
        self.config = config
        self.val_config = val_config

        vocab = Vocab()
        vocab.load(config.word2id_path, config.id2word_path)
        self.vocab = vocab
        self.config.vocab_size = vocab.vocab_size

        # To initialize simulated conversations
        self.start_sentences = self.load_sentences(self.config.dataset_dir)
        self.eval_data = self.get_data_loader(train=False)
        self.build_models()

        if self.config.load_rl_ckpt:
            self.load_models()

        self.set_up_optimizers()
        self.set_up_summary()
        self.set_up_logging()

        if self.config.rl_batch_size == self.config.beam_size:
            raise ValueError('Decoding breaks if batch_size == beam_size')

    def build_models(self):
        config = deepcopy(self.config)

        # If loading RL checkpoint, don't try to load the ckpt through Solver
        if self.config.load_rl_ckpt:
            config.checkpoint = None

        if self.config.model != 'VHRED':
            raise ValueError("Only VHRED currently supported")

        print('Building policy network...')
        self.policy_net = VariationalSolver(
            config, None, self.eval_data, vocab=self.vocab, is_train=False)
        self.policy_net.build()

        print('Building simulator network...')
        self.simulator_net = VariationalSolver(
            config, None, self.eval_data, vocab=self.vocab, is_train=False)
        self.simulator_net.build()
        self.simulator_net.model.eval()

        print('Successfully initialized policy and simulator networks')

    def set_up_optimizers(self):
        self.optimizers = {}
        named_params = list(self.policy_net.model.named_parameters())

        if self.config.vhrl or self.config.reinforce:
            manager_worker_params = [p for name, p in named_params]
            self.optimizers['worker_manager'] = optim.Adam(manager_worker_params,
                                                           lr=self.config.learning_rate)
        elif self.config.decoupled_vhrl:
            # No gradients flow from worker to manager if decoupled
            manager_params = [p for name, p in named_params if 'decoder' not in name]
            worker_params = [p for name, p in named_params if 'decoder' in name]
            self.optimizers['manager'] = optim.Adam(manager_params,
                                                    lr=self.config.manager_learning_rate)
            self.optimizers['worker'] = optim.Adam(worker_params,
                                                   lr=self.config.worker_learning_rate)

    def train(self):
        """Function to initiate RL training loop.
        """
        if self.config.vhrl:
            print('**Starting VHRL training!**\n'
                  f'Will make {self.config.num_steps} joint manager-worker updates')
        elif self.config.reinforce:
            print('**Starting REINFORCE training!**\n'
                  f'Will make {self.config.num_steps} worker updates')
        elif self.config.decoupled_vhrl:
            print('**Starting DECOUPLED VHRL training!**\n'
                  f'Will make {self.config.num_steps} alternating manager-worker updates')
        else:
            raise ValueError('Training mode not understood. '
                             'Choose from --vhrl | --reinforce | --decoupled_vhrl')
        print('... \n ... \n ...')

        # Starting RL training loop
        while self.step <= self.config.num_steps:
            self.step += 1
            self.train_step()

            if self.step % self.config.print_every_n == 0:
                self.print_summary()

            if self.step % self.config.log_every_n == 0:
                self.write_summary(self.step)

            if self.step % self.config.save_every_n == 0:
                self.save_model(self.step)

    def train_step(self):
        """RL training step.
        Behavior depends on mode --vhrl | --reinforce | --decoupled_vhrl
        """
        conversations, manager_actions, worker_actions = self.run_episodes()
        rewards = self.compute_rewards(conversations, self.config.rewards,
                                       self.config.reward_weights,
                                       self.config.gamma)

        # Collect some logging info
        self.manager_actions_history.append(manager_actions.mean().item())
        self.worker_actions_history.append(worker_actions.mean().item())
        response_lens = [len(resp.split()) for conv in conversations for resp in conv[1::2]]
        self.response_len = np.mean(response_lens)
        self.recent_dialog = conversations[0]

        if self.config.vhrl or self.config.reinforce:
            optimizer = self.optimizers['worker_manager']
            alpha, beta = self.config.alpha, self.config.beta

        elif self.config.decoupled_vhrl:
            # Update manager on even steps, update worker on odd steps
            if self.step % 2 == 0:
                update_turn = 'manager'
                alpha, beta = self.config.alpha, 0
            else:
                update_turn = 'worker'
                alpha, beta = 0, self.config.beta

            # Reset optimizer moving averages as model parametrs changed
            # by different optimizer in previous step
            optimizer = self.optimizers[update_turn]
            optimizer.state = defaultdict(dict)

        # We do not distinguish between manager and worker level rewards
        # but this could be added by calculating two different sets of rewards
        loss = (-1 * (alpha * manager_actions
                     + beta * worker_actions) * rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.model.parameters(),
                                        self.config.gradient_clip)
        optimizer.step()

    def run_episodes(self):
        """Simulate conversations which are interactions for RL training.

        Note we use s+l-1 indexing since the conversations are flattened into
        sentences. To find the output corresponding to a certain conversation
        we need to find where its sentences start (at index s-1), then we find the
        last sentence in that conversation (at l additional steps). We extract
        the output corresponding to that sentence.
        """
        # Initialize interactions with sentence from train set
        conversations = deepcopy(random.sample(self.start_sentences, self.config.rl_batch_size))
        manager_actions = []
        worker_actions = []

        simulated_turn = False
        # episode_len actions implies 2 * episode_len + 1 turns
        for turn in range(2 * self.config.episode_len):
            batch = self.policy_net.batchify(conversations)
            sentences = batch[0]
            sentence_length = batch[1]
            conversation_length = batch[2]

            net = self.simulator_net if simulated_turn else self.policy_net
            output = net.model(sentences, sentence_length,
                               conversation_length, [],
                               rl_mode=True, decode=True)

            # Index corresponding to the start of each conversation
            start_idx = np.cumsum([0] + conversation_length.tolist()[:-1])
            responses = [output[0][s+l-1].tolist()
                         for s, l in zip(start_idx, conversation_length.tolist())]
            decoded = [self.policy_net.vocab.decode(resp) for resp in responses]
            _ = [conv.append(decoded[i]) for i, conv in enumerate(conversations)]

            if not simulated_turn:
                # Get worker actions
                response_len = self.get_response_len(responses)
                word_probs = [output[6][s+l-1]
                              for s, l in zip(start_idx, conversation_length.tolist())]
                log_p_words = [torch.sum(torch.log(word_probs[i][:l]))
                               for i, l in enumerate(response_len)]

                # Divide by len to eliminate preference for longer responses
                log_p_words = torch.stack(log_p_words) / torch.cuda.FloatTensor(response_len)
                worker_actions.append(log_p_words)

                # Get manager actions
                log_p_z = torch.stack(
                    [output[7][s+l-1, ]
                        for s, l in zip(start_idx, conversation_length.tolist())])
                manager_actions.append(log_p_z)

            # Switch speaker
            simulated_turn = not simulated_turn

        return conversations, torch.stack(manager_actions, dim=1), torch.stack(worker_actions, dim=1)

    def compute_rewards(self, conversations, rewards_lst, reward_weights, gamma=0.0):
        supported = {'reward_question', 'reward_you', 'reward_toxicity',
                     'reward_bot_deepmoji', 'reward_user_deepmoji',
                     'reward_conversation_repetition', 'reward_utterance_repetition',
                     'reward_infersent_coherence', 'reward_deepmoji_coherence',
                     'reward_word2vec_coherence', 'reward_bot_response_length',
                     'reward_word_similarity', 'reward_USE_similarity'}

        episode_len = self.config.episode_len
        num_convs = self.config.rl_batch_size
        combined_rewards = np.zeros((num_convs, episode_len))

        for r, w in zip(rewards_lst, reward_weights):
            if r not in supported: raise NotImplementedError()
            reward_func = getattr(hrl_rewards, r)
            rewards = reward_func(conversations)
            discounted = discount(rewards, gamma)
            normalized = normalizeZ(discounted)
            combined_rewards += float(w) * normalized

            self.rewards_history[r].append(rewards.mean().item())

        # [num_convs, num_actions] = [rl_batch_size, episode_len]
        return to_var(torch.FloatTensor(combined_rewards))

    def get_response_len(self, responses):
        lens = []
        for resp in responses:
            try:
                lens.append(resp.index(EOS_ID)+1)
            except ValueError:
                lens.append(self.config.max_unroll)
        return lens

    def set_up_summary(self):
        # Initialize counters and summaries
        self.step = 0
        self.manager_actions_history = [0]
        self.worker_actions_history = [0]
        self.rewards_history = {}
        for r in self.config.rewards:
            self.rewards_history[r] = [0]

    def print_summary(self):
        print(20 * '*')
        print(20 * '*')
        # print(40 * '*')
        print(f'Summary at update step {self.step}')
        for r in self.config.rewards:
            print(r + ': ')
            print('\t Batch: {:.3f} \n'
                  '\t Running: {:.3f}'.format(
                        self.rewards_history[r][-1],
                        np.mean(self.rewards_history[r])))
            print(10 * '-')

        print('log utterance probability (worker actions):')
        print('\t Batch: {:.3f} \n'
              '\t Running: {:.3f}'.format(
                    self.worker_actions_history[-1],
                    np.mean(self.worker_actions_history)))
        print(10 * '-')
        print('log PDF of z (manager actions):')
        print('\t Batch: {:.3f} \n'
              '\t Running: {:.3f}'.format(
                    self.manager_actions_history[-1],
                    np.mean(self.manager_actions_history)))
        print(10 * '-')
        print('Response length:')
        print('\t Batch: {:.0f}'.format(self.response_len))
        print(10 * '-')
        print('Sample dialog:')
        print(self.recent_dialog)
        print(20 * '*')
        print(20 * '*')

        sys.stdout.flush()

    def write_summary(self, t):
        metrics_dict = {'step': self.step,
                        'response_len': self.response_len,
                        'sample_dialog': self.recent_dialog,
                        'batch_manager_action': self.manager_actions_history[-1],
                        'running_manager_actions': np.mean(self.manager_actions_history),
                        'batch_worker_action': self.worker_actions_history[-1],
                        'running_worker_actions': np.mean(self.worker_actions_history)
                        }

        for r in self.config.rewards:
            metrics_dict['batch-' + r] = self.rewards_history[r][-1]
            metrics_dict['running-' + r] = np.mean(self.rewards_history[r])

        for metric, val in metrics_dict.items():
            if metric not in {'step', 'sample_dialog'}:
                self.writer.update_loss(
                    loss=val,
                    step_i=t,
                    name=metric)

        # Write pandas csv with metrics to save dir
        self.df = self.df.append(metrics_dict, ignore_index=True)
        self.df.to_csv(self.pandas_path)

    def set_up_logging(self):
        # Get save path
        time_now = datetime.now().strftime('%Y-%m-%d_%H;%M;%S')
        default_save_path = Path('model_checkpoints/rl/')

        # Folder for type of RL used
        if self.config.reinforce:
            rl_algorithm = 'reinforce'
        elif self.config.vhrl:
            rl_algorithm = 'vhrl'
        elif self.config.decoupled_vhrl:
            rl_algorithm = 'decoupled_vhrl'

        # Folder for type of rewards used
        extra_save_dir = self.config.extra_save_dir
        if not extra_save_dir:
            if len(self.config.rewards) == 1:
                extra_save_dir = self.config.rewards[0]
            else:
                reward_names = [r[len('reward_'):] for r in self.config.rewards]
                extra_save_dir = 'reward_' + '-'.join(reward_names)

        # Make save path
        self.save_dir = default_save_path.joinpath(
            self.config.data, rl_algorithm, extra_save_dir, self.policy_net.config.model, time_now)

        # Make directory and save config
        print("Saving output to", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'config.txt'), 'w') as f:
            print(self.config, file=f)

        # Make loggers
        self.writer = TensorboardWriter(self.save_dir)
        self.pandas_path = os.path.join(self.save_dir, "metrics.csv")
        self.df = pd.DataFrame()

    def save_model(self, t):
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.save_dir, f'policy_net{t}.pkl')

        print('%'*5)
        print('%'*5)
        print(f'Saving parameters to {ckpt_path}')
        print('%'*5)
        print('%'*5)

        torch.save(self.policy_net.model.state_dict(), ckpt_path)

    def load_models(self):
        """Load parameters from RL checkpoint"""
        # Override specific checkpoint with particular one
        base_checkpoint_dir = str(Path(self.config.checkpoint).parent)
        if self.config.rl_ckpt_epoch is not None:
            policy_ckpt_path = os.path.join(
                base_checkpoint_dir, 'policy_net' + str(
                self.config.rl_ckpt_epoch) + '.pkl')
            self.iter = int(self.config.rl_ckpt_epoch)
        else:
            ckpt_file = self.config.checkpoint.replace(base_checkpoint_dir, '')
            ckpt_file = ckpt_file.replace('/', '')
            ckpt_num = ckpt_file[len('policy_net'):ckpt_file.find('.')]
            policy_ckpt_path = self.config.checkpoint
            self.iter = int(ckpt_num)

        print(f'Loading parameters for policy net from {policy_ckpt_path}')
        policy_ckpt = torch.load(policy_ckpt_path)
        policy_ckpt = convert_old_checkpoint_format(policy_ckpt)
        self.policy_net.model.load_state_dict(policy_ckpt)

        # Ensure weights are initialized to be on the GPU when necessary
        if torch.cuda.is_available():
            print('Converting checkpointed model to cuda tensors')
            self.policy_net.model.cuda()

    def get_data_loader(self, train=True):
        if train:
            sentences_path = self.config.sentences_path
            conversation_length_path = self.config.conversation_length_path
            sentence_length_path = self.config.sentence_length_path
            batch_size = self.config.rl_batch_size

        else:
            sentences_path = self.val_config.sentences_path
            conversation_length_path = self.val_config.conversation_length_path
            sentence_length_path = self.val_config.sentence_length_path
            batch_size = self.config.batch_size

        return get_loader(
            sentences=load_pickle(sentences_path),
            conversation_length=load_pickle(conversation_length_path),
            sentence_length=load_pickle(sentence_length_path),
            vocab=self.vocab,
            batch_size=batch_size)

    def load_sentences(self, data_dir):
        """Function that loads start sentences from train data.
        Used for starting simulated conversations.
        """
        sent_dir = data_dir.joinpath('train/raw_sentences.pkl')
        conversations = pickle.load(open(sent_dir, 'rb'))
        sentences = [[conv[0]] for conv in conversations
                     if len(conv[0].split()) > 5 and len(conv[0].split()) < 20]
        return sentences

    def interact(self):
        print("Commencing interaction with bot trained with RL")
        self.policy_net.interact(
            max_sentence_length=self.config.max_sentence_length,
            max_conversation_length=self.config.max_conversation_length,
            sample_by='priority', debug=False, print_history=True)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
