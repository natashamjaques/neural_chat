""" This code implements Q-learning from a static batch of data with KL-control,
monte-carlo target Q-value estimation, Psi-learning, and model averaging."""
import sys
import os
import copy
import pickle
import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import count
import random
from math import isnan
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from model.solver import Solver, VariationalSolver
from model.models import VariationalModels
from model.utils import to_var, pad, EOS_ID, TensorboardWriter, convert_old_checkpoint_format
from model.data_loader import get_loader
import replay_buffer
import rewards


class BatchQ:
    def __init__(self, config, val_config):
        self.config = config
        self.val_config = val_config

        # Load experience replay buffer from file
        self.experience = replay_buffer.CsvReplayBuffer(
            config.experience_path, raw=config.raw_buffer, 
            history_len=config.max_conversation_length, config=config,
            max_sentence_length=config.max_sentence_length, 
            rewards=config.rewards, reward_weights=config.reward_weights,
            model_averaging=config.model_averaging)
        self.vocab = self.experience.vocab
        self.config.vocab_size = self.experience.vocab.vocab_size
        self.action_dim = self.experience.vocab.vocab_size

        # Check that all required rewards are in the buffer; if not, compute
        for r in config.rewards:
            if r not in self.experience.buffer.columns.values:
                reward_func = getattr(rewards, r)
                self.experience = reward_func(self.experience)

        # Build internal hierarchical models
        self.eval_data = self.get_data_loader()
        self.build_models()
        
        if self.config.load_rl_ckpt:
            self.load_models()

        self.q_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.q_net.model.parameters()),
            lr=self.config.learning_rate)

        self.set_up_logging()

    def q_update(self):
        """General Q learning update."""
        # Sample a batch
        batch = self.experience.sample(self.config.rl_batch_size)

        # Run underlying q network to get q value of each word in each
        # conversation in the batch. Use the same data to run the prior network
        # and get the rewards based on KL divergence from the prior.
        q_values, prior_rewards = self.get_q_values(batch)

        # Compute target Q values. These will include the rewards observed in
        # the batch (i.e. r + done * gamma * max_a' Q_T(a,s'))
        with torch.no_grad():
            target_q_values = self.get_target_q_values(batch, prior_rewards)

        loss_func = getattr(F, self.config.q_loss_func)
        loss = loss_func(q_values, target_q_values)

        assert not isnan(loss.item())
        self.q_loss_batch_history.append(loss.item())

        # Optimize the model
        self.q_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients - absolutely crucial
        torch.nn.utils.clip_grad_value_(self.q_net.model.parameters(), 
                                        self.config.gradient_clip)

        self.q_optimizer.step()

        # Update Target Networks
        tau = self.config.target_update_rate 
        for param, target_param in zip(self.q_net.model.parameters(), 
                                       self.target_q_net.model.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)

    def get_q_values(self, batch):
        """update where states are whole conversations which
        each have several sentences, and actions are a sentence (series of 
        words). Q values are per word. Target Q values are over the next word 
        in the sentence, or, if at the end of the sentence, the first word in a
        new sentence after the user response.
        """
        actions = to_var(torch.LongTensor(batch['action']))  # [batch_size]

        # Prepare inputs to Q network
        conversations = [np.concatenate(
            (conv, np.atleast_2d(batch['action'][i]))) 
            for i, conv in enumerate(batch['state'])]
        sent_lens = [np.concatenate(
            (lens, np.atleast_1d(batch['action_lens'][i]))) 
            for i, lens in enumerate(batch['state_lens'])]
        target_conversations = [conv[1:] for conv in conversations]
        conv_lens = [len(c) - 1 for c in conversations]
        if self.config.model not in VariationalModels:
            conversations = [conv[:-1] for conv in conversations]
            sent_lens = np.concatenate([l[:-1] for l in sent_lens])
        else:
            sent_lens = np.concatenate([l for l in sent_lens])
        conv_lens = to_var(torch.LongTensor(conv_lens))
        
        # Run Q network. Will produce [num_sentences, max sent len, vocab size]
        all_q_values = self.run_seq2seq_model(
            self.q_net, conversations, sent_lens, target_conversations, 
            conv_lens)

        # Index to get only q values for actions taken (last sentence in each 
        # conversation)
        start_q = torch.cumsum(torch.cat(
            (to_var(conv_lens.data.new(1).zero_()), conv_lens[:-1])), 0)
        conv_q_values = torch.stack(
            [all_q_values[s+l-1, :, :]
             for s, l in zip(start_q.data.tolist(), conv_lens.data.tolist())], 
             0)  # [num_sentences, max_sent_len, vocab_size]

        # Limit by actual sentence length (remove padding) and flatten into
        # long list of words
        word_q_values = torch.cat(
            [conv_q_values[i, :l, :] 
             for i, l in enumerate(batch['action_lens'])], 
             0)  # [total words, vocab_size]
        word_actions = torch.cat(
            [actions[i, :l] for i, l in enumerate(batch['action_lens'])], 
            0)  # [total words]

        # Extract q values corresponding to actions taken
        q_values = word_q_values.gather(
            1, word_actions.unsqueeze(1)).squeeze()  # [total words]

        """ Compute KL metrics """
        prior_rewards = None

        # Get probabilities from policy network
        q_dists = torch.nn.functional.softmax(word_q_values, 1)
        q_probs = q_dists.gather(
                1, word_actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            # Run pretrained prior network. 
            # [num_sentences, max sent len, vocab size]
            all_prior_logits = self.run_seq2seq_model(
                self.pretrained_prior, conversations, sent_lens, 
                target_conversations, conv_lens)

            # Get relevant actions. [num_sentences, max_sent_len, vocab_size]
            conv_prior = torch.stack(
                [all_prior_logits[s+l-1, :, :]
                for s, l in zip(
                    start_q.data.tolist(), conv_lens.data.tolist())], 0)  
            
            # Limit by actual sentence length (remove padding) and flatten.
            # [total words, vocab_size]
            word_prior_logits = torch.cat(
                [conv_prior[i, :l, :] 
                for i, l in enumerate(batch['action_lens'])], 0)

            # Take the softmax
            prior_dists = torch.nn.functional.softmax(
                word_prior_logits, 1)

            kl_div = F.kl_div(q_dists.log(), prior_dists, reduce=False)

            # [total words]
            prior_probs = prior_dists.gather(
                1, word_actions.unsqueeze(1)).squeeze()
            logp_logq = prior_probs.log() - q_probs.log()

            if self.config.model_averaging:
                model_avg_sentences = batch['model_averaged_probs']

                # Convert to tensors and flatten into [num_words]
                word_model_avg = torch.cat([to_var(
                    torch.FloatTensor(m)) for m in model_avg_sentences], 0)
                
                # Compute KL from model-averaged prior
                prior_rewards = word_model_avg.log() - q_probs.log()

                # Clip because KL should never be negative, so because we 
                # are subtracting KL, rewards should never be positive
                prior_rewards = torch.clamp(prior_rewards, max=0.0)
            
            elif self.config.kl_control and self.config.kl_calc == 'integral':
                # Note: we reward the negative KL divergence to ensure the 
                # RL model stays close to the prior
                prior_rewards = -1.0 * torch.sum(kl_div, dim=1)
            elif self.config.kl_control:
                prior_rewards = logp_logq

            if self.config.kl_control:
                prior_rewards = prior_rewards * self.config.kl_weight_c
                self.kl_reward_batch_history.append(
                    torch.sum(prior_rewards).item())

            # Track all metrics
            self.kl_div_batch_history.append(torch.mean(kl_div).item())
            self.logp_batch_history.append(torch.mean(prior_probs.log()).item())
            self.logp_logq_batch_history.append(torch.mean(logp_logq).item())

        return q_values, prior_rewards

    def get_target_q_values(self, batch, prior_rewards=None):
        rewards = to_var(torch.FloatTensor(batch['rewards']))  # [batch_size]
        not_done = to_var(torch.FloatTensor(1 - batch['done']))  # [batch_size]
        self.sampled_reward_batch_history.append(torch.sum(rewards).item())

        # Prepare inputs to target Q network. Append a blank sentence to get
        # best response at next utterance to user input. (Next state 
        # includes user input).
        blank_sentence = np.zeros((1, self.config.max_sentence_length))
        next_state_convs = [np.concatenate(
            (conv, blank_sentence)) for conv in batch['next_state']]
        next_state_lens = [np.concatenate(
            (lens, [1])) for lens in batch['next_state_lens']]
        next_targets = [conv[1:] for conv in next_state_convs]
        next_conv_lens = [len(c) - 1 for c in next_state_convs]
        if self.config.model not in VariationalModels:
            next_state_convs = [conv[:-1] for conv in next_state_convs]
            next_state_lens = np.concatenate(
                [l[:-1] for l in next_state_lens])
        else:
            next_state_lens = np.concatenate([l for l in next_state_lens])
        next_conv_lens = to_var(torch.LongTensor(next_conv_lens))

        # [monte_carlo_count, num_sentences, max sent len, vocab size]
        _mc_target_q_values = [[]] * self.config.monte_carlo_count
        for t in range(self.config.monte_carlo_count):
            # Run target Q network. Output is size:
            # [num_sentences, max sent len, vocab size]
            if self.config.monte_carlo_count == 1:
                # In this setting, we don't use dropout out at inference time at all
                all_target_q_values = self.run_seq2seq_model(
                    self.target_q_net, next_state_convs, next_state_lens,
                    next_targets, next_conv_lens)
            else:
                # In this setting, each time we draw a new dropout mask (at inference time)
                all_target_q_values = self.run_seq2seq_model(
                    self.target_q_net, next_state_convs, next_state_lens,
                    next_targets, next_conv_lens)

            # Target indexing: last sentence is a blank to get value of next
            # response. Second last is the user response. 3rd last is models own
            # actions. Note that targets begin at the 2nd word of each sentence.
            start_t = torch.cumsum(torch.cat(
                (to_var(next_conv_lens.data.new(1).zero_()),
                    next_conv_lens[:-1])), 0)
            conv_target_q_values = torch.stack(
                [all_target_q_values[s+l-3, 1:, :]
                for s, l in zip(start_t.data.tolist(),
                                next_conv_lens.data.tolist())],
                0)  # Dimension [num_sentences, max_sent_len - 1, vocab_size]

            # At the end of a sentence, want value of starting a new response
            # after user's response. So index into first word of last blank
            # sentence that was appended to the end of the conversation.
            next_response_targets = torch.stack(
                [all_target_q_values[s+l-1, 0, :]
                for s, l in zip(start_t.data.tolist(),
                                next_conv_lens.data.tolist())], 0)
            next_response_targets = torch.reshape(
                next_response_targets,
                [self.config.rl_batch_size, 1, -1]
                ) # [num_sentences, 1, vocab_size]
            conv_target_q_values = torch.cat(
                [conv_target_q_values, next_response_targets],
                1)  # [num_sentences, max_sent_len, vocab_size]

            # Limit target Q values by conversation length
            limit_conv_targets = [conv_target_q_values[i, :l, :]
                for i, l in enumerate(batch['action_lens'])]

            if self.config.psi_learning:
                # Target is r + gamma * log sum_a' exp(Q_target(s', a'))
                conv_max_targets = [torch.distributions.utils.log_sum_exp(c)
                    for c in limit_conv_targets]
                target_q_values = torch.cat(
                    [rewards[i] + not_done[i] * self.config.gamma * c.squeeze()
                     for i, c in enumerate(conv_max_targets)], 0)  # [total words]
            else:
                # Target is r + gamma * max_a' Q_target(s',a'). Reward and done are
                # at the level of conversation, so add and multiply in before
                # flattening and taking max.
                word_target_q_values = torch.cat(
                    [rewards[i] + not_done[i] * self.config.gamma * c
                     for i, c in enumerate(limit_conv_targets)],
                    0)  # [total words, vocab_size]
                target_q_values, _ = word_target_q_values.max(1)

            _mc_target_q_values[t] = target_q_values
        mc_target_q_values = torch.stack(_mc_target_q_values, 0)

        min_target_q_values, _ = mc_target_q_values.min(0)

        if self.config.kl_control:
            min_target_q_values += prior_rewards

        return min_target_q_values

    def q_learn(self):
        self.q_loss_history = []
        self.q_loss_batch_history = []
        self.sampled_reward_history = []
        self.sampled_reward_batch_history = []
        if self.config.kl_control:
            self.kl_reward_history = []
            self.kl_reward_batch_history = []
        
        # Need to track KL metrics even for baselines for plots
        self.kl_div_history = []
        self.kl_div_batch_history = []
        self.logp_history = []
        self.logp_batch_history = []
        self.logp_logq_history = []
        self.logp_logq_batch_history = []

        print('Commencing training at step', self.t)
        while self.t <= self.config.num_steps:
            self.q_update()

            # Log metrics
            if self.t % self.config.log_every_n == 0:
                self.epoch_q_loss = np.sum(self.q_loss_batch_history) \
                    / self.config.log_every_n
                self.q_loss_history.append(self.epoch_q_loss)
                self.q_loss_batch_history = []
                print('Average Q loss at step', self.t, '=', self.epoch_q_loss)

                self.epoch_sampled_reward = np.sum(
                    self.sampled_reward_batch_history) / self.config.log_every_n
                self.sampled_reward_history.append(self.epoch_sampled_reward)
                self.sampled_reward_batch_history = []
                print('\tAverage sampled batch reward =', 
                    self.epoch_sampled_reward)
                
                if self.config.kl_control:
                    self.epoch_kl_reward = np.sum(
                        self.kl_reward_batch_history) \
                        / self.config.log_every_n
                    self.kl_reward_history.append(
                        self.epoch_kl_reward)
                    self.kl_reward_batch_history = []
                    print('\tAverage data prior reward =', 
                        self.epoch_kl_reward)

                # Logging KL for plots
                self.epoch_kl_div = np.sum(
                    self.kl_div_batch_history) / self.config.log_every_n
                self.kl_div_history.append(self.epoch_kl_div)
                self.kl_div_batch_history = []
                self.epoch_logp = np.sum(
                    self.logp_batch_history) / self.config.log_every_n
                self.logp_history.append(self.epoch_logp)
                self.logp_batch_history = []
                self.epoch_logp_logq = np.sum(
                    self.logp_logq_batch_history) / self.config.log_every_n
                self.logp_logq_history.append(self.epoch_logp_logq)
                self.logp_logq_batch_history = []
                
                sys.stdout.flush()
                self.write_summary(self.t)

            if self.t > 0 and self.t % self.config.save_every_n == 0:
                self.save_model(self.t)

            self.t += 1

    def build_models(self):
        config = copy.deepcopy(self.config)

        # If loading RL checkpoint, ensure it doesn't try to load the ckpt 
        # through Solver
        if self.config.load_rl_ckpt:
            config.checkpoint = None  

        if self.config.model in VariationalModels:
            self.q_net = VariationalSolver(
                config, None, self.eval_data, vocab=self.vocab, is_train=True)
            self.target_q_net = VariationalSolver(
                config, None, self.eval_data, vocab=self.vocab, is_train=True)
        else:
            self.q_net = Solver(
                config, None, self.eval_data, vocab=self.vocab, is_train=True)
            self.target_q_net = Solver(
                config, None, self.eval_data, vocab=self.vocab, is_train=True)
        print('Building Q network')
        self.q_net.build()

        print('\nBuilding Target Q network')
        self.target_q_net.build()

        if self.config.model in VariationalModels:
            self.pretrained_prior = VariationalSolver(
                self.config, None, self.eval_data, vocab=self.vocab, 
                is_train=True)
        else:
            self.pretrained_prior = Solver(
                self.config, None, self.eval_data, vocab=self.vocab, 
                is_train=True)
        print('Building prior network')
        self.pretrained_prior.build()
            
        # Freeze the weights of the prior so it stays constant
        self.pretrained_prior.model.eval()
        for params in self.pretrained_prior.model.parameters():
            params.requires_grad = False

        print('Successfully initialized Q networks')
        self.t = 0

    def run_seq2seq_model(self, q_net, input_conversations, sent_lens, 
                     target_conversations, conv_lens):
        # Prepare the batch
        sentences = [sent for conv in input_conversations for sent in conv]
        targets = [sent for conv in target_conversations for sent in conv]

        if not (np.all(np.isfinite(sentences)) 
                and np.all(np.isfinite(targets)) 
                and np.all(np.isfinite(sent_lens))):
            print("Input isn't finite")

        sentences = to_var(torch.LongTensor(sentences))
        targets = to_var(torch.LongTensor(targets))
        sent_lens = to_var(torch.LongTensor(sent_lens))

        # Run Q network
        q_outputs = q_net.model(sentences, sent_lens, conv_lens, targets, 
                                rl_mode=True)
        return q_outputs[0]  # [num_sentences, max_sentence_len, vocab_size]

    def write_summary(self, t):
        metrics_to_log = ['epoch_q_loss', 'epoch_sampled_reward', 
                          'epoch_kl_div', 'epoch_logp', 'epoch_logp_logq']

        if self.config.kl_control:
            metrics_to_log.append('epoch_kl_reward')

        metrics_dict = {}
        for metric in metrics_to_log:
            met_val = getattr(self, metric, None)
            metrics_dict[metric] = met_val
            if met_val is not None:
                self.writer.update_loss(
                    loss=met_val,
                    step_i=t,
                    name=metric)

        # Write pandas csv with metrics to save dir
        self.df = self.df.append(metrics_dict, ignore_index=True)
        self.df.to_csv(self.pandas_path)

    def set_up_logging(self):
        # Get save path
        time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        default_save_path = Path(
                os.path.expanduser('~') + '/dialog/model_checkpoints/rl/')

        # Folder for type of RL used
        experiment_name = self.config.experiment_name
        if experiment_name is None:
            # if self.config.double_q: experiment_name = 'double_q'
            if self.config.model_averaging:
                experiment_name = 'model_averaging'
            elif self.config.kl_control: 
                experiment_name = 'kl_control'
                if self.config.kl_calc == 'sample': experiment_name += '_sample'
            else: experiment_name = 'batch_q'
            if self.config.psi_learning: 
                experiment_name += '/psi_learning'
            if self.config.monte_carlo_count > 1:
                experiment_name = 'monte_carlo_targets/' + experiment_name

        # Folder for type of rewards used
        extra_save_dir = self.config.extra_save_dir
        if not extra_save_dir:
            if len(self.config.rewards) == 1:
                extra_save_dir = self.config.rewards[0]
            else:
                extra_save_dir = 'reward_combo'

        # Folder for which model was used
        extra_model_desc = ""
        if self.config.context_input_only:
            extra_model_desc = 'input_only_'
        if self.config.emotion and 'input_only' not in extra_model_desc: 
            extra_model_desc += "emotion_"
        if self.config.infersent and 'input_only' not in extra_model_desc: 
            extra_model_desc += "infersent_"

        # Make save path
        self.save_dir = default_save_path.joinpath(
            self.q_net.config.data, extra_save_dir, experiment_name,
            extra_model_desc + self.q_net.config.model, time_now)

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
        ckpt_path = os.path.join(self.save_dir, f'q_net{t}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.q_net.model.state_dict(), ckpt_path)

        ckpt_path = os.path.join(self.save_dir, f'target_q_net{t}.pkl')
        torch.save(self.target_q_net.model.state_dict(), ckpt_path)

    def load_models(self):
        """Load parameters from RL checkpoint"""
        # Override specific checkpoint with particular one
        base_checkpoint_dir = str(Path(self.config.checkpoint).parent)
        if self.config.rl_ckpt_epoch is not None:
            q_ckpt_path = os.path.join(
                base_checkpoint_dir, 'q_net' + str(
                self.config.rl_ckpt_epoch) + '.pkl')
            target_q_ckpt_path = os.path.join(
                base_checkpoint_dir, 'target_q_net' + str(
                self.config.rl_ckpt_epoch) + '.pkl')
            self.t = int(self.config.rl_ckpt_epoch)
        else:
            ckpt_file = self.config.checkpoint.replace(base_checkpoint_dir, '')
            ckpt_file = ckpt_file.replace('/', '')
            ckpt_num = ckpt_file[len('q_net'):ckpt_file.find('.')]
            q_ckpt_path = self.config.checkpoint
            target_q_ckpt_path = os.path.join(
                base_checkpoint_dir, 'target_q_net' + ckpt_num + '.pkl')
            self.t = int(ckpt_num)

        print(f'Loading parameters for Q net from {q_ckpt_path}')
        q_ckpt = torch.load(q_ckpt_path)
        q_ckpt = convert_old_checkpoint_format(q_ckpt)
        self.q_net.model.load_state_dict(q_ckpt)

        print(f'Loading parameters for target Q net from {target_q_ckpt_path}')
        target_q_ckpt = torch.load(target_q_ckpt_path)
        target_q_ckpt = convert_old_checkpoint_format(target_q_ckpt)
        self.target_q_net.model.load_state_dict(target_q_ckpt)

        # Ensure weights are initialized to be on the GPU when necessary
        if torch.cuda.is_available():
            print('Converting checkpointed model to cuda tensors')
            self.q_net.model.cuda()
            self.target_q_net.model.cuda()

    def get_data_loader(self):
        # If checkpoint is for an emotion model, load that pickle file
        emotion_sentences = None
        if self.config.emotion:
            emotion_sentences = load_pickle(self.val_config.emojis_path)

        # Load infersent embeddings if necessary
        infersent_sentences = None
        if self.config.infersent:
            print('Loading infersent sentence embeddings...')
            infersent_sentences = load_pickle(self.val_config.infersent_path)
            embedding_size = infersent_sentences[0][0].shape[0]
            self.config.infersent_output_size = embedding_size
            self.val_config.infersent_output_size = embedding_size

        return get_loader(
            sentences=load_pickle(self.val_config.sentences_path),
            conversation_length=load_pickle(self.val_config.conversation_length_path),
            sentence_length=load_pickle(self.val_config.sentence_length_path),
            vocab=self.vocab,
            batch_size=self.config.batch_size,
            emojis=emotion_sentences,
            infersent=infersent_sentences)

    def interact(self):
        print("Commencing interaction with bot trained with RL")
        self.q_net.interact(
                max_sentence_length=self.config.max_sentence_length, 
                max_conversation_length=self.config.max_conversation_length, 
                sample_by='priority', debug=True, print_history=True)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
