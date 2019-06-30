"""Implements a discrete version of Batch Constrained Q learning."""
import argparse
import os
import pickle
import copy
import torch
import numpy as np

from model.configs import get_config_from_dir
from model.utils import Vocab, convert_old_checkpoint_format, detokenize, to_var
from model.solver import Solver, VariationalSolver
from model.data_loader import get_loader
from model.models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_config_args():
    parser = argparse.ArgumentParser()
    
    # Must provide checkpoints of pre-trained models to load
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--q_checkpoint', type=str, default=None)

    parser.add_argument('--bcq_n', type=int, default=10, 
                        help="Number of samples from prior in BCQ algorithm.")

    return vars(parser.parse_args())


class DBCQ:
    def __init__(self, prior_config, rl_config, beam_size=5):
        self.prior_config = prior_config
        self.rl_config = rl_config
        self.rl_config.beam_size = beam_size

        print('Loading Vocabulary...')
        self.vocab = Vocab()
        self.vocab.load(prior_config.word2id_path, prior_config.id2word_path)
        self.prior_config.vocab_size = self.vocab.vocab_size
        self.rl_config.vocab_size = self.vocab.vocab_size
        print(f'Vocabulary size: {self.vocab.vocab_size}')
        
        self.eval_data = self.get_data_loader()
        self.build_models()

    def build_models(self):
        rl_config = copy.deepcopy(self.rl_config)
        rl_config.checkpoint = None  

        print('Building Q network')
        if rl_config.model in VariationalModels:
            self.q_net = VariationalSolver(
                rl_config, None, self.eval_data, vocab=self.vocab, 
                is_train=False)
        else:
            self.q_net = Solver(
                rl_config, None, self.eval_data, vocab=self.vocab, 
                is_train=False)
        self.q_net.build()
        self.load_q_network()

        print('Building prior network')
        if self.prior_config.model in VariationalModels:
            self.pretrained_prior = VariationalSolver(
                self.prior_config, None, self.eval_data, vocab=self.vocab, 
                is_train=False)
        else:
            self.pretrained_prior = Solver(
                self.prior_config, None, self.eval_data, vocab=self.vocab, 
                is_train=False)
        self.pretrained_prior.build()
       
        # Freeze the weights so they stay constant
        self.pretrained_prior.model.eval()
        for params in self.pretrained_prior.model.parameters():
            params.requires_grad = False
        self.q_net.model.eval()
        for params in self.q_net.model.parameters():
            params.requires_grad = False

    def load_q_network(self):
        """Load parameters from RL checkpoint"""
        print(f'Loading parameters for Q net from {self.rl_config.checkpoint}')
        q_ckpt = torch.load(self.rl_config.checkpoint)
        q_ckpt = convert_old_checkpoint_format(q_ckpt)
        self.q_net.model.load_state_dict(q_ckpt)

        # Ensure weights are initialized to be on the GPU when necessary
        if torch.cuda.is_available():
            print('Converting checkpointed model to cuda tensors')
            self.q_net.model.cuda()

    def get_data_loader(self):
        # If checkpoint is for an emotion model, load that pickle file
        emotion_sentences = None
        if self.prior_config.emotion:
            emotion_sentences = load_pickle(self.prior_config.emojis_path)

        # Load infersent embeddings if necessary
        infersent_sentences = None
        if self.prior_config.infersent:
            print('Loading infersent sentence embeddings...')
            infersent_sentences = load_pickle(self.prior_config.infersent_path)
            embedding_size = infersent_sentences[0][0].shape[0]
            self.prior_config.infersent_output_size = embedding_size

        return get_loader(
            sentences=load_pickle(self.prior_config.sentences_path),
            conversation_length=load_pickle(
                self.prior_config.conversation_length_path),
            sentence_length=load_pickle(self.prior_config.sentence_length_path),
            vocab=self.vocab,
            batch_size=self.prior_config.batch_size,
            emojis=emotion_sentences,
            infersent=infersent_sentences)

    def interact(self, max_conversation_length=5, sample_by='priority', 
                 debug=True):
        model_name = self.prior_config.model
        context_sentences = []

        print("Time to start a conversation with the chatbot! It's name is", 
              model_name)
        username = input("What is your name? ")

        print("Let's start chatting. You can type 'quit' at any time to quit.")
        utterance = input("Input: ")
        print("\033[1A\033[K")  # Erases last line of output

        while (utterance.lower() != 'quit' and utterance.lower() != 'exit'):
            # Process utterance
            sentences = utterance.split('/')

            # Code and decode user input to show how it is transformed for model
            coded, lens = self.pretrained_prior.process_user_input(sentences)
            decoded = [self.vocab.decode(sent) for i, sent in enumerate(
                coded) if i < lens[i]]
            print(username + ':', '. '.join(decoded))

            # Append to conversation
            context_sentences.extend(sentences)

            gen_response = self.generate_response_to_input(
                context_sentences, max_conversation_length, sample_by=sample_by,
                debug=debug)

            # Append generated sentences to conversation
            context_sentences.append(gen_response)

            # Print and get next user input
            print("\n" + model_name + ": " + gen_response)
            utterance = input("Input: ")
            print("\033[1A\033[K")

    def process_raw_text_into_input(self, raw_text_sentences,
                                    max_conversation_length=5, debug=False,):
        sentences, lens = self.pretrained_prior.process_user_input(
            raw_text_sentences, self.rl_config.max_sentence_length)

        # Remove any sentences of length 0
        sentences = [sent for i, sent in enumerate(sentences) if lens[i] > 0]
        good_raw_sentences = [sent for i, sent in enumerate(
            raw_text_sentences) if lens[i] > 0]
        lens = [l for l in lens if l > 0]

        # Trim conversation to max length
        sentences = sentences[-max_conversation_length:]
        lens = lens[-max_conversation_length:]
        good_raw_sentences = good_raw_sentences[-max_conversation_length:]
        convo_length = len(sentences)

        # Convert to torch variables
        input_sentences = to_var(torch.LongTensor(sentences))
        input_sentence_length = to_var(torch.LongTensor(lens))
        input_conversation_length = to_var(torch.LongTensor([convo_length]))

        if debug:
            print('\n**Conversation history:**')
            for sent in sentences:
                print(self.vocab.decode(list(sent)))

        return (input_sentences, input_sentence_length, 
                input_conversation_length)

    def duplicate_context_for_beams(self, sentences, sent_lens, conv_lens,
                                    beams):
        conv_lens = conv_lens.repeat(len(beams))
        # [beam_size * sentences, sentence_len]
        if len(sentences) > 1:
            targets = torch.cat(
                [torch.cat([sentences[1:,:], beams[i,:].unsqueeze(0)], 0) 
                for i in range(len(beams))], 0)
        else:
            targets = beams

        # HRED
        if self.rl_config.model not in VariationalModels:
            sent_lens = sent_lens.repeat(len(beams))
            return sentences, sent_lens, conv_lens, targets
        
        # VHRED, VHCR
        new_sentences = torch.cat(
            [torch.cat([sentences, beams[i,:].unsqueeze(0)], 0) 
            for i in range(len(beams))], 0)
        new_len = to_var(torch.LongTensor([self.rl_config.max_sentence_length]))
        sent_lens = torch.cat(
            [torch.cat([sent_lens, new_len], 0) for i in range(len(beams))])
        return new_sentences, sent_lens, conv_lens, targets

    def generate_response_to_input(self, raw_text_sentences, 
                                   max_conversation_length=5,
                                   sample_by='priority', emojize=True,
                                   debug=True):
        with torch.no_grad():
            (input_sentences, input_sent_lens, 
             input_conv_lens) = self.process_raw_text_into_input(
                raw_text_sentences, debug=debug,
                max_conversation_length=max_conversation_length)

            # Initialize a tensor for beams
            beams = to_var(torch.LongTensor(
                np.ones((self.rl_config.beam_size, 
                        self.rl_config.max_sentence_length))))

            # Create a batch with the context duplicated for each beam
            (sentences, sent_lens, 
            conv_lens, targets) = self.duplicate_context_for_beams(
                input_sentences, input_sent_lens, input_conv_lens, beams)

            # Continuously feed beam sentences into networks to sample the next 
            # best word, add that to the beam, and continue
            for i in range(self.rl_config.max_sentence_length):
                # Run both models to obtain logits
                prior_output = self.pretrained_prior.model(
                    sentences, sent_lens, conv_lens, targets, rl_mode=True)
                all_prior_logits = prior_output[0]
                
                q_output = self.q_net.model(
                    sentences, sent_lens, conv_lens, targets, rl_mode=True)
                all_q_logits = q_output[0]

                # Select only those logits for next word
                q_logits = all_q_logits[:, i, :].squeeze()
                prior_logits = all_prior_logits[:, i, :].squeeze()

                # Get prior distribution for next word in each beam
                prior_dists = torch.nn.functional.softmax(prior_logits, 1)

                for b in range(self.rl_config.beam_size):
                    # Sample from the prior bcq_n times for each beam
                    dist = torch.distributions.Categorical(prior_dists[b,:])
                    sampled_idxs = dist.sample_n(self.rl_config.bcq_n)

                    # Select sample with highest q value
                    q_vals = torch.stack(
                        [q_logits[b, idx] for idx in sampled_idxs])
                    _, best_word_i = torch.max(q_vals, 0) 
                    best_word = sampled_idxs[best_word_i]

                    # Update beams
                    beams[b, i] = best_word

                (sentences, sent_lens, 
                 conv_lens, targets) = self.duplicate_context_for_beams(
                    input_sentences, input_sent_lens, input_conv_lens, beams)
            
            generated_sentences = beams.cpu().numpy()

        if debug:
            print('\n**All generated responses:**')
            for gen in generated_sentences:
                print(detokenize(self.vocab.decode(list(gen))))
        
        gen_response = self.pretrained_prior.select_best_generated_response(
            generated_sentences, sample_by, beam_size=self.rl_config.beam_size)

        decoded_response = self.vocab.decode(list(gen_response))
        decoded_response = detokenize(decoded_response)

        if emojize:
            inferred_emojis = self.pretrained_prior.botmoji.emojize_text(
                raw_text_sentences[-1], 5, 0.07)
            decoded_response = inferred_emojis + " " + decoded_response
        
        return decoded_response


if __name__ == '__main__':

    kwargs_dict = parse_config_args()

    kwargs_dict['mode'] = 'valid'
    prior_config = get_config_from_dir(kwargs_dict['checkpoint'], **kwargs_dict)

    kwargs_dict['load_rl_ckpt'] = True
    rl_config = get_config_from_dir(kwargs_dict['q_checkpoint'], **kwargs_dict)

    dbcq = DBCQ(prior_config, rl_config)
    dbcq.interact()