from itertools import cycle
import getpass
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from math import isnan
import re
import math
import pickle
import gensim
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok

import model.models as models
from model.layers import masked_cross_entropy, cross_entropy_soft_targets, mean_squared_error
from model.utils import Tokenizer, to_var, time_desc_decorator, TensorboardWriter, pad_and_pack, normal_kl_div, to_bow, \
    bag_of_words_loss, normal_kl_div, embedding_metric, pad_tokens, UNK_ID, EOS_ID, detokenize, \
    convert_old_checkpoint_format
from model.utils.novel_metrics import novel_metrics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from torchMoji.examples.botmoji import Botmoji
from inferSent.botsent import Botsent

word2vec_path = os.path.join(ROOT_DIR, "datasets/GoogleNews-vectors-negative300.bin")

# Tokenizer
tokenizer = Tokenizer('spacy')

# Params to save to output csv file
OUTPUT_FILE_PARAMS = ['context_input_only', 'context_size', 'data',
                      'decoder_hidden_size', 'dropout', 'embedding_size',
                      'emo_weight', 'emotion', 'encoder_hidden_size',
                      'infersent', 'infersent_weight', 'kl_annealing_iter',
                      'model', 'sentence_drop', 'word_drop', 'tie_embedding']


class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model
        self.botmoji = Botmoji()
        self.botsent = Botsent(config.dataset_dir.joinpath('train'), version=1, explained_var=0.95)
        self.detokenizer = Detok()

        # Info for saving epoch metrics to a csv file
        if self.config.mode == 'train':
            self.pandas_path = os.path.join(config.save_path, "metrics.csv")
            self.outfile_dict = {k: getattr(config, k) for k in OUTPUT_FILE_PARAMS}
            self.df = pd.DataFrame()

        self.save_priming_sentences()

    def save_priming_sentences(self):
        # sample a random conversation, if needed for priming
        if self.eval_data_loader is not None:
            self.priming_sample_idx = np.random.choice(np.arange(len(self.eval_data_loader)))
        self.bot_priming_sentences = ["I am just a bot, but I sometimes give great advice!",
                                      "I have big plans for today; I'm going to wait until someone starts chatting with me!",
                                      "I can't experience the weather or the physical world, but I can understand when you express being excited or annoyed by it!",
                                      "I love my job. It is all about talking to interesting people and learning from them!",
                                      "Some people may think I am stuck in a box, but I actually like it. Not having a body is working out fine for me.",
                                      "Some say it is hard to do nothing. But I do exactly nothing the days that noone talks to me.",
                                      "I think talking to people is both the easiest and the hardest thing to do.",
                                      "I may not experience what you experience, but I try to do the best I can in listening to you and understanding you."]
        np.random.shuffle(self.bot_priming_sentences)

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

            # orthogonal initialiation for hidden weights
            # input gate bias for GRUs
            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initialization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                    # bias_hh is concatenation of reset, input, new gates
                    # only set the input gate bias to 2.0
                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        # Overview Parameters
        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            self.optimizer = self.config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate)

    def save_model(self, epoch):
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')
        if not self.config.load_rl_ckpt:
            epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
            self.epoch_i = int(epoch)
        else:
            base_checkpoint_dir = str(Path(checkpoint).parent)
            ckpt_file = checkpoint.replace(base_checkpoint_dir, '')
            ckpt_file = ckpt_file.replace('/', '')
            self.epoch_i = int(ckpt_file[len('q_net'):ckpt_file.find('.')])
        ckpt = torch.load(checkpoint)

        # Maintain backwards compatibility of checkpoints
        ckpt = convert_old_checkpoint_format(ckpt)

        self.model.load_state_dict(ckpt)

        # Ensure weights are initialized to be on the GPU when necessary
        if torch.cuda.is_available():
            print('Converting checkpointed model to cuda tensors')
            self.model.cuda()

    def write_summary(self, epoch_i):
        metrics = ['epoch_loss', 'epoch_recon_loss', 'epoch_kl_div', 'kl_mult',
                   'epoch_bow_loss', 'epoch_emoji_loss', 'epoch_infersent_loss',
                   'validation_loss', 'validation_emoji_loss',
                   'validation_infersent_loss', 'epoch_embedding_average',
                   'epoch_embedding_extrema', 'epoch_embedding_greedy']
        metrics_dict = {}
        for metric in metrics:
            met_val = getattr(self, metric, None)
            metrics_dict[metric] = met_val
            if met_val is not None:
                self.writer.update_loss(
                    loss=met_val,
                    step_i=epoch_i + 1,
                    name=metric)

        # Write pandas csv with metrics to save dir
        metrics_dict.update(self.outfile_dict)
        self.df = self.df.append(metrics_dict, ignore_index=True)
        self.df.to_csv(self.pandas_path)

    @time_desc_decorator('Training Start!')
    def train(self):
        epoch_loss_history = []
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            self.model.train()
            results = {}
            results['batch_loss_history'] = []
            results['emoji_loss_history'] = []
            results['infersent_loss_history'] = []
            results['n_total_words'] = 0
            results['n_total_emojis'] = 0
            results['n_total_infersent'] = 0
            for batch_i, batch_data in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # reset gradient
                self.optimizer.zero_grad()

                results = self._process_batch(batch_i, batch_data, results,
                                              mode='train', decode=False)

                batch_loss = results['batch_loss']
                if self.config.emotion:
                    batch_loss += self.config.emo_weight * results['emo_loss']
                if self.config.infersent:
                    batch_loss += self.config.infersent_weight * results['infersent_loss']

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()

                if batch_i % self.config.print_every == 0:
                    norm_batch_loss = results['batch_loss'].item() / results['n_words'].item()
                    print_str = f'Epoch: {epoch_i + 1}, iter {batch_i}: loss = {norm_batch_loss:.3f}'
                    if self.config.emotion:
                        norm_emo_loss = results['emo_loss'].item() / results['n_sentences']
                        print_str += f', emoji_loss = {norm_emo_loss:.3f}'
                    if self.config.infersent:
                        norm_infersent_loss = results['infersent_loss'].item() / results['n_sentences_infersent']
                        print_str += f', infersent_loss = {norm_infersent_loss:.3f}'
                    tqdm.write(print_str)

            epoch_loss = np.sum(results['batch_loss_history']) / results['n_total_words']
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i + 1} loss average: {epoch_loss:.3f}'
            if self.config.emotion:
                emo_loss = np.sum(
                    results['emoji_loss_history']) * self.config.emo_weight
                self.epoch_emoji_loss = emo_loss / results['n_total_emojis']
                print_str += f', emoji_loss = {self.epoch_emoji_loss:.3f}'
            if self.config.infersent:
                infersent_loss = np.sum(
                    results['infersent_loss_history']) * self.config.infersent_weight
                self.epoch_infersent_loss = infersent_loss / results['n_total_infersent']
                print_str += f', infersent_loss = {self.epoch_infersent_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if self.config.evaluate_embedding_metrics:
                (self.epoch_embedding_average, self.epoch_embedding_extrema,
                 self.epoch_embedding_greedy) = self.embedding_metric()

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def _process_batch(self, batch_i, batch_data, results, decode=False, mode='val'):
        # conversations: (batch_size) list of conversations
        #   conversation: list of sentences
        #   sentence: list of tokens
        # conversation_length: list of int
        # sentence_length: (batch_size) list of conversation list of sentence_lengths
        conversations = batch_data[0]
        conversation_length = batch_data[1]
        sentence_length = batch_data[2]

        input_conversations = [conv[:-1] for conv in conversations]
        target_conversations = [conv[1:] for conv in conversations]

        # flatten input and target conversations
        input_sentences = [sent for conv in input_conversations for sent in conv]
        target_sentences = [sent for conv in target_conversations for sent in conv]
        input_sentence_length = [l for len_list in sentence_length for l in len_list[:-1]]
        target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
        input_conversation_length = [l - 1 for l in conversation_length]

        if self.config.emotion:
            emoji_conversations = batch_data[3]
            input_emoji_conversations = [conv[:-1] for conv in emoji_conversations]
            emoji_sentences = [sent for conv in input_emoji_conversations for sent in conv]
            if mode == 'train':
                emoji_sentences = to_var(torch.FloatTensor(emoji_sentences))
            else:
                with torch.no_grad():
                    emoji_sentences = to_var(torch.FloatTensor(emoji_sentences))
            results['n_sentences'] = emoji_sentences.shape[0]
            results['n_total_emojis'] += results['n_sentences']

        if self.config.infersent:
            infersent_conversations = batch_data[4]
            input_infersent_conversations = [conv[:-1] for conv in infersent_conversations]
            infersent_sentences = [sent for conv in input_infersent_conversations for sent in conv]
            if mode == 'train':
                infersent_sentences = to_var(torch.FloatTensor(infersent_sentences))
            else:
                with torch.no_grad():
                    infersent_sentences = to_var(torch.FloatTensor(infersent_sentences))
            results['n_sentences_infersent'] = infersent_sentences.shape[0]
            results['n_total_infersent'] += results['n_sentences_infersent']

        extra_context_inputs = None
        if self.config.context_input_only:
            extra_context_inputs = torch.cat((emoji_sentences, infersent_sentences), 1)

        if mode == 'train':
            input_sentences = to_var(torch.LongTensor(input_sentences))
            target_sentences = to_var(torch.LongTensor(target_sentences))
            input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
            target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
            input_conversation_length = to_var(
                torch.LongTensor(input_conversation_length))
        else:
            with torch.no_grad():
                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))

        if mode == 'val' and batch_i == 0:
            self.generate_sentence(input_sentences,
                                   input_sentence_length,
                                   input_conversation_length,
                                   target_sentences,
                                   extra_context_inputs=extra_context_inputs)

        preds = self.model(
            input_sentences,
            input_sentence_length,
            input_conversation_length,
            target_sentences,
            decode=decode,
            extra_context_inputs=extra_context_inputs)
        sentence_logits = preds[0]

        results['batch_loss'], results['n_words'] = masked_cross_entropy(
            sentence_logits,
            target_sentences,
            target_sentence_length)

        if self.config.emotion:
            emoji_preds = preds[1]
            results['emo_loss'] = cross_entropy_soft_targets(emoji_preds,
                                                             emoji_sentences)
            results['emoji_loss_history'].append(results['emo_loss'].item())

        if self.config.infersent:
            infersent_preds = preds[2]
            results['infersent_loss'] = mean_squared_error(infersent_preds,
                                                           infersent_sentences)
            results['infersent_loss_history'].append(
                results['infersent_loss'].item())

        assert not isnan(results['batch_loss'].item())
        results['batch_loss_history'].append(results['batch_loss'].item())
        results['n_total_words'] += results['n_words'].item()

        return results

    def generate_sentence(self, input_sentences, input_sentence_length,
                          input_conversation_length, target_sentences,
                          extra_context_inputs=None):
        self.model.eval()

        # [batch_size, max_seq_len, vocab_size]
        preds = self.model(
            input_sentences,
            input_sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True,
            extra_context_inputs=extra_context_inputs)
        generated_sentences = preds[0]

        # write output to file
        with open(os.path.join(self.config.save_path, 'samples.txt'), 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(
                    input_sentences, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                print(s)
            print('')

    def evaluate(self):
        results_dict = self._evaluate(mode='val')
        epoch_loss = results_dict['epoch_loss']
        print_str = f'Validation loss: {epoch_loss:.3f}\n'
        if self.config.emotion:
            emo_loss = np.sum(
                results_dict['emoji_loss_history']) * self.config.emo_weight
            self.validation_emoji_loss = emo_loss / results_dict['n_total_emojis']
            print_str += f', emoji_loss = {self.validation_emoji_loss:.3f}'
        if self.config.infersent:
            infsent_loss = np.sum(
                results_dict['infersent_loss_history']) * self.config.infersent_weight
            self.validation_infersent_loss = infsent_loss / results_dict['n_total_infersent']
            print_str += f', infersent_loss = {self.validation_infersent_loss:.6f}'
        print(print_str)

        return epoch_loss

    def test(self):
        results_dict = self._evaluate(mode='test')
        epoch_loss = results_dict['epoch_loss']
        n_total_words = results_dict['n_total_words']

        print(f'Number of words: {n_total_words}')
        print(f'Bits per word: {epoch_loss:.3f}')
        word_perplexity = np.exp(epoch_loss)

        if self.config.emotion:
            emo_loss = np.sum(results_dict['emoji_loss_history'])
            emoji_epoch_loss = emo_loss / results_dict['n_total_emojis']
            print(f'Emoji loss: {emoji_epoch_loss:.3f}')
        if self.config.infersent:
            infsent_loss = np.sum(results_dict['infersent_loss_history'])
            infersent_epoch_loss = infsent_loss / results_dict['n_total_infersent']
            print(f'Infersent_loss = {infersent_epoch_loss:.6f}')

        print_str = f'Word perplexity: {word_perplexity:.3f}\n'
        print(print_str)

        return word_perplexity

    def _evaluate(self, mode='val'):
        self.model.eval()
        results = {}
        results['batch_loss_history'] = []
        results['emoji_loss_history'] = []
        results['infersent_loss_history'] = []
        results['n_total_words'] = 0
        results['n_total_emojis'] = 0
        results['n_total_infersent'] = 0
        for batch_i, batch_data in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            results = self._process_batch(batch_i, batch_data, results,
                                          decode=False, mode=mode)

        results['epoch_loss'] = np.sum(results['batch_loss_history']) / results['n_total_words']
        return results

    def process_user_input(self, sentences, max_sentence_length=30):
        # Add periods if they are missing
        for i in range(len(sentences)):
            sentences[i] = sentences[i].rstrip()  # Remove trailing whitespace
            if len(sentences[i]) > 0 and sentences[i][-1] not in ['.', '!', '?']:
                sentences[i] = sentences[i] + '.'

        tokenized = [tokenizer(sent) for sent in sentences]
        trimmed = [tokens[:max_sentence_length - 1] for tokens in tokenized]
        lens = [len(t) + 1 for t in trimmed]  # Add 1 for <eos> token
        padded = [pad_tokens(tokens) for tokens in trimmed]
        coded = self.eval_data_loader.dataset.sent2id(padded)
        return coded, lens

    def self_play(self, conversation_length=10, max_sentence_length=30, max_conversation_length=5,
                  sample_by='priority', print_history=False):
        context_sentences = ['']

        for i in range(conversation_length):
            gen_response = self.generate_response_to_input(
                context_sentences, max_conversation_length, max_sentence_length,
                sample_by=sample_by)

            # Append generated sentences to conversation
            context_sentences.append(gen_response)

        if print_history:
            for s in context_sentences:
                print(s)
        # removing the first empty prompt
        return context_sentences[1:]

    def interact(self, max_sentence_length=30, max_conversation_length=5,
                 sample_by='priority', debug=False, print_history=False):
        model_name = self.config.model
        context_sentences = []

        print("Time to start a conversation with the chatbot! It's name is", model_name)
        username = input("What is your name? ")

        print("Let's start chatting. You can type 'quit' at any time to quit.")
        utterance = input("Input: ")
        print("\033[1A\033[K")  # Erases last line of output

        while (utterance.lower() != 'quit' and utterance.lower() != 'exit'):
            # Process utterance
            sentences = utterance.split('/')

            # Code and decode user input to show how it is transformed for model
            coded, lens = self.process_user_input(sentences,
                                                  max_sentence_length)
            decoded = [self.vocab.decode(sent) for i, sent in enumerate(coded) if i < lens[i]]
            print(username + ':', '. '.join(decoded))

            # Append to conversation
            context_sentences.extend(sentences)

            gen_response = self.generate_response_to_input(
                context_sentences, max_conversation_length, max_sentence_length,
                sample_by=sample_by, debug=debug)

            # Append generated sentences to conversation
            context_sentences.append(gen_response)

            # Print and get next user input
            print("\n" + model_name + ": " + gen_response)
            utterance = input("Input: ")
            print("\033[1A\033[K")

        if print_history:
            for s in context_sentences: print(s)

    def process_raw_text_into_input(self, raw_text_sentences, 
                                    max_sentence_length=30, 
                                    max_conversation_length=5, debug=False,
                                    priming_condition=0):
        sentences, lens = self.process_user_input(raw_text_sentences, 
                                                  max_sentence_length)

        # Remove any sentences of length 0
        sentences = [sent for i, sent in enumerate(sentences) if lens[i] > 0]
        good_raw_sentences = [sent for i, sent in enumerate(raw_text_sentences) if lens[i] > 0]
        lens = [l for l in lens if l > 0]

        # Pad in extra sentences from the real data if the conversation is too
        # short to stop the model from spazzing out.
        if self.config.data == 'reddit_casual' and len(sentences) < max_conversation_length:
            # priming with a set of fixed sentences
            if priming_condition == 1:
                (self.priming_sentences,
                 self.priming_sentences_lens) = self.process_user_input(
                    self.bot_priming_sentences, max_sentence_length)
                sentences = self.priming_sentences + sentences
                lens = self.priming_sentences_lens + lens
            # priming with a sample convo from eval dataset
            elif priming_condition == 2:
                sample = self.eval_data_loader.dataset[self.priming_sample_idx]
                sentences = sample[0] + sentences
                lens = sample[2] + lens
        # else if priming_condition==0 (no priming)

        # Trim conversation to max length
        sentences = sentences[-max_conversation_length:]
        lens = lens[-max_conversation_length:]
        good_raw_sentences = good_raw_sentences[-max_conversation_length:]
        convo_length = len(sentences)

        if self.config.model in models.VariationalModels:
            # Pad a fake target sentence in because code breaks without one :/
            good_raw_sentences = good_raw_sentences + [good_raw_sentences[-1]]
            sentences = sentences + [sentences[-1]]
            lens = lens + [lens[-1]]

        extra_context_inputs = None
        if self.config.context_input_only:
            # Use botsent and botmoji raw outputs
            infersent_sentences = to_var(torch.FloatTensor(
                [self.botsent.encode(s) for s in good_raw_sentences]))
            emoji_sentences = to_var(torch.FloatTensor(
                [self.botmoji.encode(s) for s in good_raw_sentences]))
            extra_context_inputs = torch.cat(
                (emoji_sentences, infersent_sentences), 1)

        # Convert to torch variables
        input_sentences = to_var(torch.LongTensor(sentences))
        input_sentence_length = to_var(torch.LongTensor(lens))
        input_conversation_length = to_var(torch.LongTensor([convo_length]))

        if debug:
            print('\n**Conversation history:**')
            for sent in sentences:
                print(self.vocab.decode(list(sent)))

        return (input_sentences, input_sentence_length, 
                input_conversation_length, extra_context_inputs)

    def generate_response_to_input(self, raw_text_sentences, 
                                   max_conversation_length=5,
                                   max_sentence_length=30, debug=False, 
                                   emojize=False, sample_by='priority', 
                                   priming_condition=0):
        (input_sentences, input_sentence_length, input_conversation_length, 
         extra_context_inputs) = self.process_raw_text_into_input(
             raw_text_sentences, max_sentence_length=max_sentence_length, 
             max_conversation_length=max_conversation_length, 
             debug=debug, priming_condition=priming_condition)
        
        output = self.model(input_sentences,
                            input_sentence_length,
                            input_conversation_length,
                            [],
                            decode=True,
                            extra_context_inputs=extra_context_inputs)
        generated_sentences = output[0]
        generated_sentences = generated_sentences.view(-1, 30).cpu().numpy()

        if debug:
            print('\n**All generated responses:**')
            for gen in generated_sentences:
                print(detokenize(self.vocab.decode(list(gen))))
        
        gen_response = self.select_best_generated_response(
            generated_sentences, sample_by)

        decoded_response = self.vocab.decode(list(gen_response))
        decoded_response = detokenize(decoded_response)

        if emojize:
            inferred_emojis = self.botmoji.emojize_text(
                raw_text_sentences[-1], 5, 0.07)
            decoded_response = inferred_emojis + " " + decoded_response
        
        return decoded_response
    
    def select_best_generated_response(self, generated_sentences, 
                                       sample_by='priority', beam_size=None):
        """ Args: 
                generated_sentences: arrays of word IDs to be decoded
                sample_by: can be None for no sampling, 'priority', or 'length'.
        """
        if beam_size is None:
            beam_size = self.config.beam_size
        generated_sentences = generated_sentences[-beam_size:,:]

        if len(generated_sentences) == 1:
            return generated_sentences[0]
        
        # Ensure each sentence has an end of sentence token.
        for i, gen in enumerate(generated_sentences):
            if EOS_ID not in gen:
                generated_sentences[i] = np.concatenate((gen[:-1], 
                                                        np.array([EOS_ID])))

        if len(generated_sentences) > 1:
            # Select the best response heuristically - long with few unks
            # discard sentences with 2 or more <unk> tokens
            few_unks = [gen for gen in generated_sentences
                        if (gen == UNK_ID).sum() < 3 and len(
                            gen) > 0 and gen[0] != EOS_ID]
            if len(few_unks) > 1:
                if sample_by == 'priority':
                    priority_idx = np.arange(len(few_unks))[::-1]
                else:
                    priority_idx = np.array([list(gen).index(EOS_ID) + 1 for gen in few_unks])
                norm = priority_idx / priority_idx.sum()
                pos = np.random.choice(np.arange(len(priority_idx)), p=norm)
                gen_response = few_unks[pos]
            elif len(few_unks) > 0:
                gen_response = few_unks[0]
            else:
                pos = np.random.choice(len(generated_sentences))
                gen_response = generated_sentences[pos]

        return gen_response

    def embedding_metric(self):
        word2vec = getattr(self, 'word2vec', None)
        if word2vec is None:
            print('Loading word2vec model')
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            self.word2vec = word2vec
        keys = word2vec.vocab
        self.model.eval()
        n_context = self.config.n_context
        n_sample_step = self.config.n_sample_step
        metric_average_history = []
        metric_extrema_history = []
        metric_greedy_history = []
        context_history = []
        sample_history = []
        n_sent = 0
        n_conv = 0
        for batch_i, batch_data in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths
            conversations = batch_data[0]
            conversation_length = batch_data[1]
            sentence_length = batch_data[2]

            conv_indices = [i for i in range(len(conversations)) if len(conversations[i]) >= n_context + n_sample_step]
            context = [c for i in conv_indices for c in [conversations[i][:n_context]]]
            ground_truth = [c for i in conv_indices for c in [conversations[i][n_context:n_context + n_sample_step]]]
            sentence_length = [c for i in conv_indices for c in [sentence_length[i][:n_context]]]

            with torch.no_grad():
                context = to_var(torch.LongTensor(context))
                sentence_length = to_var(torch.LongTensor(sentence_length))

            extra_context_inputs = None
            if self.config.context_input_only:
                emoji_conversations = batch_data[3]
                infersent_conversations = batch_data[4]
                emojis_context = [c for i in conv_indices for c in emoji_conversations[i][:n_context]]
                infersent_context = [c for i in conv_indices for c in infersent_conversations[i][:n_context]]
                with torch.no_grad():
                    emojis_input = to_var(torch.FloatTensor(emojis_context))
                    infersent_input = to_var(torch.FloatTensor(infersent_context))
                    extra_context_inputs = torch.cat(
                        (emojis_input, infersent_input), 1)

            samples = self.model.generate(
                context, sentence_length, n_context,
                extra_context_inputs=extra_context_inputs, botmoji=self.botmoji,
                botsent=self.botsent, vocab=self.vocab)

            context = context.data.cpu().numpy().tolist()
            samples = samples.data.cpu().numpy().tolist()
            context_history.append(context)
            sample_history.append(samples)

            samples = [[self.vocab.decode(sent) for sent in c] for c in samples]
            ground_truth = [[self.vocab.decode(sent) for sent in c] for c in ground_truth]

            samples = [sent for c in samples for sent in c]
            ground_truth = [sent for c in ground_truth for sent in c]

            samples = [[word2vec[s] for s in sent.split() if s in keys] for sent in samples]
            ground_truth = [[word2vec[s] for s in sent.split() if s in keys] for sent in ground_truth]

            indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != [] and g != []]
            samples = [samples[i] for i in indices]
            ground_truth = [ground_truth[i] for i in indices]
            n = len(samples)
            n_sent += n

            metric_average = embedding_metric(samples, ground_truth, word2vec, 'average')
            metric_extrema = embedding_metric(samples, ground_truth, word2vec, 'extrema')
            metric_greedy = embedding_metric(samples, ground_truth, word2vec, 'greedy')
            metric_average_history.append(metric_average)
            metric_extrema_history.append(metric_extrema)
            metric_greedy_history.append(metric_greedy)

        epoch_average = np.mean(np.concatenate(metric_average_history), axis=0)
        epoch_extrema = np.mean(np.concatenate(metric_extrema_history), axis=0)
        epoch_greedy = np.mean(np.concatenate(metric_greedy_history), axis=0)

        print('n_sentences:', n_sent)
        print_str = f'Metrics - Average: {epoch_average:.3f}, Extrema: {epoch_extrema:.3f}, Greedy: {epoch_greedy:.3f}'
        print(print_str)
        print('\n')

        return epoch_average, epoch_extrema, epoch_greedy

    def novel_metrics(self):
        """
        This function calculates the metrics related to emotion and sentence embedding.
        :return:
        epoch_emotions: the cosine distance from emotion embeddings
        epoch_infersent_pca: the cosine distance from  pca'd infersent sentence embeddings
        epoch_infersent_raw: the cosine distance from raw infersent sentence embeddings
        """
        self.model.eval()
        n_context = self.config.n_context
        n_sample_step = self.config.n_sample_step
        metric_emotion_history = []
        metric_infersent_pca_history = []
        metric_infersent_raw_history = []
        n_sent = 0
        for batch_i, batch_data in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths
            # emoji_conversations: (batch_size) list of conversations' emojis
            # infersent_pca_conversations: (batch_size) list of conversations' infersent PCA'd embedding
            conversations = batch_data[0]
            conversation_length = batch_data[1]
            sentence_length = batch_data[2]
            emoji_conversations = batch_data[3]
            infersent_pca_conversations = batch_data[4]

            conv_indices = [i for i in range(len(conversations)) if len(conversations[i]) >= n_context + n_sample_step]
            context = [c for i in conv_indices for c in [conversations[i][:n_context]]]
            emojis_context = [c for i in conv_indices for c in emoji_conversations[i][:n_context]]
            infersent_pca_context = [c for i in conv_indices for c in infersent_pca_conversations[i][:n_context]]

            ground_truth = [c for i in conv_indices for c in [conversations[i][n_context:n_context + n_sample_step]]]
            ground_truth_emojis = [c for i in conv_indices for c in
                                   [emoji_conversations[i][n_context:n_context + n_sample_step]]]
            ground_truth_infersent_pca = [c for i in conv_indices for c in
                                      [infersent_pca_conversations[i][n_context:n_context + n_sample_step]]]

            sentence_length = [c for i in conv_indices for c in [sentence_length[i][:n_context]]]

            with torch.no_grad():
                context = to_var(torch.LongTensor(context))
                sentence_length = to_var(torch.LongTensor(sentence_length))

            extra_context_inputs = None
            if self.config.context_input_only:
                with torch.no_grad():
                    emojis_input = to_var(torch.FloatTensor(emojis_context))
                    infersent_pca_input = to_var(torch.FloatTensor(infersent_pca_context))
                    extra_context_inputs = torch.cat(
                        (emojis_input, infersent_pca_input), 1)

            samples = self.model.generate(
                context, sentence_length, n_context,
                extra_context_inputs=extra_context_inputs, botmoji=self.botmoji,
                botsent=self.botsent, vocab=self.vocab)

            samples = samples.data.cpu().numpy().tolist()

            samples = [[self.vocab.decode(sent) for sent in c] for c in samples]
            ground_truth = [[self.vocab.decode(sent) for sent in c] for c in ground_truth]

            samples = [self.detokenizer.detokenize([sent]) for c in samples for sent in c]
            ground_truth = [self.detokenizer.detokenize([sent]) for c in ground_truth for sent in c]

            ground_truth_emojis = [sent for c in ground_truth_emojis for sent in c]
            ground_truth_infersent_pca = [sent for c in ground_truth_infersent_pca for sent in c]

            indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != '' and g != '']
            samples = [samples[i] for i in indices]
            ground_truth = [ground_truth[i] for i in indices]

            samples_emojis = self.botmoji.encode_multiple(samples)
            samples_infersent_pca = self.botsent.encode_multiple(samples)
            samples_infersent_raw = self.botsent.encode_multiple(samples, override=True, override_use_pca=False)

            ground_truth_emojis = [ground_truth_emojis[i] for i in indices]
            ground_truth_infersent_pca = [ground_truth_infersent_pca[i] for i in indices]
            ground_truth_infersent_raw = self.botsent.encode_multiple(ground_truth, override=True,
                                                                      override_use_pca=False)

            n = len(samples)
            n_sent += n

            metric_emotion = novel_metrics(samples_emojis, ground_truth_emojis)
            metric_infersent_pca = novel_metrics(samples_infersent_pca, ground_truth_infersent_pca)
            metric_infersent_raw = novel_metrics(samples_infersent_raw, ground_truth_infersent_raw)
            metric_emotion_history.append(metric_emotion)
            metric_infersent_pca_history.append(metric_infersent_pca)
            metric_infersent_raw_history.append(metric_infersent_raw)

        epoch_emotion = np.mean(np.concatenate(metric_emotion_history), axis=0)
        epoch_infersent_pca = np.mean(np.concatenate(metric_infersent_pca_history), axis=0)
        epoch_infersent_raw = np.mean(np.concatenate(metric_infersent_raw_history), axis=0)

        print('n_sentences:', n_sent)
        print_str = f'Metrics - Emotion distance: {epoch_emotion:.6f}, ' \
            f'InferSent PCA distance: {epoch_infersent_pca:.6f}, ' \
            f'InferSent raw distance: {epoch_infersent_raw:.6f}'
        print(print_str)
        print('\n')

        return epoch_emotion, epoch_infersent_pca, epoch_infersent_raw


class VariationalSolver(Solver):

    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model
        self.botmoji = Botmoji()
        self.botsent = Botsent(config.dataset_dir.joinpath('train'), version=1, explained_var=0.95)
        self.detokenizer = Detok()

        # Info for saving epoch metrics to a csv file
        if self.config.mode == 'train':
            self.pandas_path = os.path.join(config.save_path, "metrics.csv")
            self.outfile_dict = {k: getattr(config, k) for k in OUTPUT_FILE_PARAMS}
            self.df = pd.DataFrame()

        self.save_priming_sentences()

    @time_desc_decorator('Training Start!')
    def train(self):
        epoch_loss_history = []
        kl_mult = 0.0
        conv_kl_mult = 0.0

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            results = {
                'batch_loss_history': [],
                'recon_loss_history': [],
                'kl_div_history': [],
                'bow_loss_history': [],
                'emoji_loss_history': [],
                'infersent_loss_history': [],
                'n_total_words': 0,
                'n_total_emojis': 0,
                'n_total_infersent': 0
            }
            self.model.train()
            # self.evaluate()

            for batch_i, batch_data in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # reset gradient
                self.optimizer.zero_grad()

                results = self._process_batch(batch_i, batch_data, results,
                                              mode='train', kl_mult=kl_mult)
                batch_loss = results['batch_loss']
                n_words = results['n_words']

                # Add additional losses to batch loss
                if self.config.bow:
                    batch_loss += results['bow_loss']
                if self.config.emotion:
                    batch_loss += results['emo_loss'] * self.config.emo_weight
                if self.config.infersent:
                    batch_loss += results['infersent_loss'] * self.config.infersent_weight

                if batch_i % self.config.print_every == 0:
                    batch_loss_norm = batch_loss.item() / n_words.item()
                    recon_loss_norm = results['recon_loss'].item() / n_words.item()
                    kl_div_norm = results['kl_div'].item() / n_words.item()

                    print_str = f'Epoch: {epoch_i + 1}, iter {batch_i}: '
                    print_str += f'loss = {batch_loss_norm:.3f}, '
                    print_str += f'recon = {recon_loss_norm:.3f}, '
                    print_str += f'kl_div = {kl_div_norm:.3f}'
                    if self.config.bow:
                        bow_loss_norm = results['bow_loss'].item() / n_words.item()
                        print_str += f', bow_loss = {bow_loss_norm:.3f}'
                    if self.config.emotion:
                        emo_loss_norm = results['emo_loss'].item() / results['n_sentences']
                        print_str += f', emoji_loss = {emo_loss_norm:.3f}'
                    if self.config.infersent:
                        infersent_loss_norm = (results['infersent_loss'].item()
                                               / results['n_sentences_infersent'])
                        print_str += f', infersent_loss = {infersent_loss_norm:.3f}'
                    tqdm.write(print_str)

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()
                kl_mult = min(kl_mult + 1.0 / self.config.kl_annealing_iter, 1.0)

            epoch_loss = np.sum(results['batch_loss_history']) / results['n_total_words']
            epoch_loss_history.append(epoch_loss)

            epoch_recon_loss = np.sum(results['recon_loss_history']) / results['n_total_words']
            epoch_kl_div = np.sum(results['kl_div_history']) / results['n_total_words']

            self.kl_mult = kl_mult
            self.epoch_loss = epoch_loss
            self.epoch_recon_loss = epoch_recon_loss
            self.epoch_kl_div = epoch_kl_div

            print_str = f'Epoch {epoch_i + 1} loss average: {epoch_loss:.3f}, '
            print_str += f'recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
            if results['bow_loss_history']:
                self.epoch_bow_loss = np.sum(results['bow_loss_history']) \
                                      / results['n_total_words']
                print_str += f', bow_loss = {self.epoch_bow_loss:.3f}'
            if self.config.emotion:
                self.epoch_emoji_loss = np.sum(results['emoji_loss_history']) \
                                        / results['n_total_emojis'] \
                                        * self.config.emo_weight
                print_str += f', emoji_loss = {self.epoch_emoji_loss:.3f}'
            if self.config.infersent:
                self.epoch_infersent_loss = np.sum(
                    results['infersent_loss_history']) \
                                            / results['n_total_infersent'] \
                                            * self.config.infersent_weight
                print_str += f', infersent_loss = {self.epoch_infersent_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if self.config.evaluate_embedding_metrics:
                (self.epoch_embedding_average, self.epoch_embedding_extrema,
                 self.epoch_embedding_greedy) = self.embedding_metric()

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

        return epoch_loss_history

    def _process_batch(self, batch_i, batch_data, results, mode='val', kl_mult=1.0):
        # conversations: (batch_size) list of conversations
        #   conversation: list of sentences
        #   sentence: list of tokens
        # conversation_length: list of int
        # sentence_length: (batch_size) list of conversation list of sentence_lengths
        conversations = batch_data[0]
        conversation_length = batch_data[1]
        sentence_length = batch_data[2]

        target_conversations = [conv[1:] for conv in conversations]

        # flatten input and target conversations
        sentences = [sent for conv in conversations for sent in conv]
        input_conversation_length = [l - 1 for l in conversation_length]
        target_sentences = [sent for conv in target_conversations for sent in conv]
        target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
        sentence_length = [l for len_list in sentence_length for l in len_list]

        if self.config.emotion:
            emoji_conversations = batch_data[3]
            input_emoji_conversations = [conv[:-1] for conv in emoji_conversations]
            emoji_sentences = [sent for conv in input_emoji_conversations for sent in conv]
            if mode == 'train':
                emoji_sentences = to_var(torch.FloatTensor(emoji_sentences))
            else:
                with torch.no_grad():
                    emoji_sentences = to_var(torch.FloatTensor(emoji_sentences))
            results['n_sentences'] = emoji_sentences.shape[0]
            results['n_total_emojis'] += results['n_sentences']

        if self.config.infersent:
            infersent_conversations = batch_data[4]
            input_infersent_conversations = [conv[:-1] for conv in infersent_conversations]
            infersent_sentences = [sent for conv in input_infersent_conversations for sent in conv]
            if mode == 'train':
                infersent_sentences = to_var(torch.FloatTensor(infersent_sentences))
            else:
                with torch.no_grad():
                    infersent_sentences = to_var(torch.FloatTensor(infersent_sentences))
            results['n_sentences_infersent'] = infersent_sentences.shape[0]
            results['n_total_infersent'] += results['n_sentences_infersent']

        extra_context_inputs = None
        if self.config.context_input_only:
            emojis_with_targets = np.array([sent for conv in emoji_conversations for sent in conv])
            infersent_with_targets = np.array([sent for conv in infersent_conversations for sent in conv])
            extra_context_inputs = np.concatenate([emojis_with_targets, infersent_with_targets], axis=1)
            if mode == 'train':
                extra_context_inputs = to_var(torch.FloatTensor(extra_context_inputs))
            else:
                with torch.no_grad():
                    extra_context_inputs = to_var(torch.FloatTensor(extra_context_inputs))

        if mode == 'val':
            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

            if batch_i == 0:
                input_conversations = [conv[:-1] for conv in conversations]
                input_sentences = [sent for conv in input_conversations for sent in conv]
                with torch.no_grad():
                    input_sentences = to_var(torch.LongTensor(input_sentences))
                self.generate_sentence(sentences,
                                       sentence_length,
                                       input_conversation_length,
                                       input_sentences,
                                       target_sentences,
                                       extra_context_inputs=extra_context_inputs)
        elif mode == 'train':
            sentences = to_var(torch.LongTensor(sentences))
            sentence_length = to_var(torch.LongTensor(sentence_length))
            input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
            target_sentences = to_var(torch.LongTensor(target_sentences))
            target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

        outputs = self.model(
            sentences,
            sentence_length,
            input_conversation_length,
            target_sentences,
            extra_context_inputs=extra_context_inputs)
        sentence_logits = outputs[0]
        results['kl_div'] = outputs[1]

        results['recon_loss'], results['n_words'] = masked_cross_entropy(
            sentence_logits,
            target_sentences,
            target_sentence_length)

        results['batch_loss'] = results['recon_loss'] + kl_mult * results['kl_div']
        results['batch_loss_history'].append(results['batch_loss'].item())
        results['recon_loss_history'].append(results['recon_loss'].item())
        results['kl_div_history'].append(results['kl_div'].item())
        results['n_total_words'] += results['n_words'].item()

        if self.config.bow:
            results['bow_loss'] = self.model.compute_bow_loss(target_conversations)
            results['bow_loss_history'].append(results['bow_loss'].item())
        if self.config.emotion:
            emoji_preds = outputs[4]
            results['emo_loss'] = cross_entropy_soft_targets(emoji_preds,
                                                             emoji_sentences)
            results['emoji_loss_history'].append(results['emo_loss'].item())
        if self.config.infersent:
            infersent_preds = outputs[5]
            results['infersent_loss'] = mean_squared_error(infersent_preds,
                                                           infersent_sentences)
            results['infersent_loss_history'].append(
                results['infersent_loss'].item())

        assert not isnan(results['batch_loss'].item())

        return results

    def generate_sentence(self, sentences, sentence_length,
                          input_conversation_length, input_sentences,
                          target_sentences, extra_context_inputs=None):
        """Generate output of decoder (single batch)"""
        self.model.eval()

        # [batch_size, max_seq_len, vocab_size]
        preds = self.model(
            sentences,
            sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True,
            extra_context_inputs=extra_context_inputs)
        generated_sentences = preds[0]

        # write output to file
        with open(os.path.join(self.config.save_path, 'samples.txt'), 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(
                    input_sentences, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                print(s)
            print('')

    def evaluate(self):
        self.model.eval()
        results = {
            'batch_loss_history': [],
            'recon_loss_history': [],
            'kl_div_history': [],
            'bow_loss_history': [],
            'emoji_loss_history': [],
            'infersent_loss_history': [],
            'n_total_words': 0,
            'n_total_emojis': 0,
            'n_total_infersent': 0
        }
        for batch_i, batch_data in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            results = self._process_batch(batch_i, batch_data, results, mode='val')

        epoch_loss = np.sum(
            results['batch_loss_history']) / results['n_total_words']
        epoch_recon_loss = np.sum(
            results['recon_loss_history']) / results['n_total_words']
        epoch_kl_div = np.sum(
            results['kl_div_history']) / results['n_total_words']

        print_str = f'Validation loss: {epoch_loss:.3f}, '
        print_str += f'recon_loss: {epoch_recon_loss:.3f}, '
        print_str += f'kl_div: {epoch_kl_div:.3f}'
        if results['bow_loss_history']:
            epoch_bow_loss = np.sum(results['bow_loss_history']) / results['n_total_words']
            print_str += f', bow_loss = {epoch_bow_loss:.3f}'
        if self.config.emotion:
            self.validation_emoji_loss = np.sum(
                results['emoji_loss_history']) / results['n_total_emojis'] \
                                         * self.config.emo_weight
            print_str += f', emoji_loss = {self.validation_emoji_loss:.3f}'
        if self.config.infersent:
            self.validation_infersent_loss = np.sum(
                results['infersent_loss_history']) \
                                             / results['n_total_infersent'] \
                                             * self.config.infersent_weight
            print_str += f', infersent_loss = {self.validation_infersent_loss:.3f}'
        print(print_str)
        print('\n')

        return epoch_loss

    def importance_sample(self):
        ''' Perform importance sampling to get tighter bound
        '''
        self.model.eval()
        weight_history = []
        emoji_loss_history = []
        infersent_loss_history = []
        n_total_words = 0
        n_total_emojis = 0
        n_total_infersent = 0
        kl_div_history = []
        for batch_i, batch_data in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths
            conversations = batch_data[0]
            conversation_length = batch_data[1]
            sentence_length = batch_data[2]

            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            sentences = [sent for conv in conversations for sent in conv]
            input_conversation_length = [l - 1 for l in conversation_length]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            sentence_length = [l for len_list in sentence_length for l in len_list]

            if self.config.emotion:
                emoji_conversations = batch_data[3]
                input_emoji_conversations = [conv[:-1] for conv in emoji_conversations]
                emoji_sentences = [sent for conv in input_emoji_conversations for sent in conv]
                with torch.no_grad():
                    emoji_sentences = to_var(torch.FloatTensor(emoji_sentences))
                n_sentences = emoji_sentences.shape[0]
                n_total_emojis += n_sentences

            if self.config.infersent:
                infersent_conversations = batch_data[4]
                input_infersent_conversations = [conv[:-1] for conv in infersent_conversations]
                infersent_sentences = [sent for conv in input_infersent_conversations for sent in conv]
                with torch.no_grad():
                    infersent_sentences = to_var(torch.FloatTensor(infersent_sentences))
                n_sentences_infersent = infersent_sentences.shape[0]
                n_total_infersent += n_sentences_infersent

            extra_context_inputs = None
            if self.config.context_input_only:
                emojis_with_targets = np.array([sent for conv in emoji_conversations for sent in conv])
                infersent_with_targets = np.array([sent for conv in infersent_conversations for sent in conv])
                extra_context_inputs = np.concatenate([emojis_with_targets, infersent_with_targets], axis=1)
                with torch.no_grad():
                    extra_context_inputs = to_var(torch.FloatTensor(extra_context_inputs))

            # n_words += sum([len([word for word in sent if word != PAD_ID]) for sent in target_sentences])
            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

            # treat whole batch as one data sample
            weights = []
            for j in range(self.config.importance_sample):
                (sentence_logits, kl_div, log_p_z, log_q_zx,
                 emoji_preds, infersent_preds) = self.model(
                    sentences,
                    sentence_length,
                    input_conversation_length,
                    target_sentences,
                    extra_context_inputs=extra_context_inputs)

                recon_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                log_w = (-recon_loss.sum() + log_p_z - log_q_zx).data
                weights.append(log_w)
                if j == 0:
                    n_total_words += n_words.item()
                    kl_div_history.append(kl_div.item())

                if self.config.emotion:
                    emo_loss = cross_entropy_soft_targets(emoji_preds, emoji_sentences)
                    emoji_loss_history.append(emo_loss.item() / n_sentences)

                if self.config.infersent:
                    infersent_loss = mean_squared_error(infersent_preds, infersent_sentences)
                    infersent_loss_history.append(infersent_loss.item() / n_sentences_infersent)

            # weights: [n_samples]
            weights = torch.stack(weights, 0)
            m = torch.floor(weights.max())
            weights = torch.log(torch.exp(weights - m).sum())
            weights = m + weights - np.log(self.config.importance_sample)
            weight_history.append(weights)

        print(f'Number of words: {n_total_words}')
        bits_per_word = -np.sum(weight_history) / n_total_words
        print(f'Bits per word: {bits_per_word:.3f}')
        word_perplexity = np.exp(bits_per_word)

        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        if self.config.emotion:
            epoch_emoji_loss = np.sum(emoji_loss_history)
            print(f'Emoji loss: {epoch_emoji_loss:.3f}')
        if self.config.infersent:
            epoch_infersent_loss = np.sum(infersent_loss_history)
            print(f'Infersent loss: {epoch_infersent_loss:.6f}')

        print_str = f'Word perplexity upperbound using {self.config.importance_sample} importance samples: {word_perplexity:.3f}, kl_div: {epoch_kl_div:.3f}\n'
        print(print_str)

        return word_perplexity