import os
import argparse
import ast
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
from model.layers.rnncells import StackedLSTMCell, StackedGRUCell

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {'cornell': data_dir.joinpath('cornell'), 'ubuntu': data_dir.joinpath('ubuntu')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
rnncell_dict = {'lstm': StackedLSTMCell, 'gru': StackedGRUCell}
username = Path.home().name
save_dir = project_dir.joinpath('model_checkpoints')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                if key == 'rnncell':
                    value = rnncell_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        dataset = self.data.lower()
        if dataset not in data_dict:
            self.dataset_dir = data_dir.joinpath(dataset)
        else:
            self.dataset_dir = data_dict[dataset]

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir.joinpath(self.mode)
        # Pickled Vocabulary
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')

        # Pickled Dataframes
        self.sentences_path = self.data_dir.joinpath('sentences.pkl')
        self.sentence_length_path = self.data_dir.joinpath('sentence_length.pkl')
        self.conversation_length_path = self.data_dir.joinpath('conversation_length.pkl')

        extra_model_desc = ""
        if self.context_input_only:
            self.emotion = True
            self.infersent = True
            extra_model_desc = 'input_only_'

        if self.emotion or self.calc_novel_embedding:
            self.emojis_path = self.data_dir.joinpath('sentence_emojis.pkl')
            if 'input_only' not in extra_model_desc: extra_model_desc += "emotion_"

        if self.infersent or self.calc_novel_embedding:
            self.infersent_path = self.data_dir.joinpath('sentence_embeddings_1_PCA_0.95.pkl')
            if 'input_only' not in extra_model_desc: extra_model_desc += "infersent_"

        # Save path
        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
            self.save_path = save_dir.joinpath(
                self.data, self.extra_save_dir, extra_model_desc + self.model,
                time_now)
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = self.save_path

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Emotion discriminator hparams
    parser.add_argument('--emotion', action='store_true')
    parser.set_defaults(emotion=False)
    parser.add_argument('--emo_activation', type=str, default='ReLU')
    parser.add_argument('--emo_embedding_size', type=int, default=128)
    parser.add_argument('--emo_num_layers', type=int, default=2)
    parser.add_argument('--emo_output_size', type=int, default=64)
    parser.add_argument('--emo_weight', type=float, default=25)

    # Embedding discriminator hparams
    parser.add_argument('--infersent', action='store_true')
    parser.set_defaults(infersent=False)
    parser.add_argument('--infersent_activation', type=str, default='ReLU')
    parser.add_argument('--infersent_embedding_size', type=int, default=128)
    parser.add_argument('--infersent_num_layers', type=int, default=2)
    parser.add_argument('--infersent_output_size', type=int, default=64)
    parser.add_argument('--infersent_weight', type=float, default=25000)

    # Use emoji and infersent as input only
    parser.add_argument('--context_input_only', action='store_true')
    parser.set_defaults(context_input_only=False)

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--eval_batch_size', type=int, default=80)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--extra_save_dir', type=str, default="")
    parser.add_argument('--evaluate_embedding_metrics', action='store_true')
    parser.set_defaults(evaluate_embedding_metrics=False)

    # Generation
    parser.add_argument('--max_unroll', type=int, default=30)
    parser.add_argument('--sample', type=str2bool, default=False,
                        help='if false, use beam search for decoding')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1)

    # Model
    parser.add_argument('--model', type=str, default='VHCR',
                        help='one of {HRED, VHRED, VHCR}')
    # Currently does not support lstm
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=500)
    parser.add_argument('--tie_embedding', type=str2bool, default=True)
    parser.add_argument('--encoder_hidden_size', type=int, default=1000)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--decoder_hidden_size', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--context_size', type=int, default=1000)
    parser.add_argument('--feedforward', type=str, default='FeedForward')
    parser.add_argument('--activation', type=str, default='Tanh')

    # VAE model
    parser.add_argument('--z_sent_size', type=int, default=100)
    parser.add_argument('--z_conv_size', type=int, default=100)
    parser.add_argument('--word_drop', type=float, default=0.0,
                        help='only applied to variational models')
    parser.add_argument('--kl_threshold', type=float, default=0.0)
    parser.add_argument('--kl_annealing_iter', type=int, default=25000)
    parser.add_argument('--importance_sample', type=int, default=100)
    parser.add_argument('--sentence_drop', type=float, default=0.0)

    # Generation
    parser.add_argument('--n_context', type=int, default=1)
    parser.add_argument('--n_sample_step', type=int, default=1)

    # BOW
    parser.add_argument('--bow', type=str2bool, default=False)

    # Utility
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    # Data
    parser.add_argument('--data', type=str, default='ubuntu')

    # Calc novel embedding metrics
    parser.add_argument('--calc_novel_embedding', type=str2bool, default=True)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


def get_config_from_dir(checkpoint_dir, **optional_kwargs):
    f = open(os.path.join(checkpoint_dir, 'config.txt'), 'r')
    lines = f.readlines()

    # Transform raw file lines into appropriate dict format
    lines = lines[1:]   # Discard line reading 'Configurations'
    lines = [l for l in lines if 'Posix' not in l]
    for i, l in enumerate(lines):
        if 'rnn' in l:
            lower = l.lower()
            if 'rnncell' in l:
                prefix = 'rnncell'
            else:
                prefix = 'rnn'
            if 'lstm' in lower:
                lines[i] = " '" + prefix + "': 'lstm',"
            else:
                lines[i] = " '" + prefix + "': 'gru',"
        if 'adam' in l:
            lines[i] = " 'optimizer': 'Adam',"

    # Transform lines into dict
    f_str = ''.join(lines)
    f_str = f_str.replace('\n', '')
    config_dict = ast.literal_eval(f_str)

    # Override with additional config args
    config_dict.update(optional_kwargs)

    # Find latest checkpoint in directory
    if (('load_rl_ckpt' in optional_kwargs and optional_kwargs['load_rl_ckpt'])
        or ('load_rl_ckpt' in config_dict and config_dict['load_rl_ckpt'])):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if '.pkl' in f]
        if 'rl_ckpt_epoch' in optional_kwargs and optional_kwargs['rl_ckpt_epoch']:
            ckpt_num = optional_kwargs['rl_ckpt_epoch']
        else:
            checkpoint_nums = []
            for c in checkpoints:
                if 'target' in c:
                    continue
                else:
                    net = c.split('_')[0] + '_net'
                    checkpoint_nums.append(int(c[len(net):c.find('.')]))
            ckpt_num = sorted(checkpoint_nums)[-1]
        latest_checkpoint = os.path.join(checkpoint_dir, net + str(ckpt_num) + '.pkl')
    else:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if str.isdigit(f[0])]
        checkpoint_nums = [int(f[:f.find('.')]) for f in checkpoints]
        latest = sorted(checkpoint_nums)[-1]
        latest_checkpoint = os.path.join(checkpoint_dir, str(latest) + '.pkl')
    print('Found latest checkpoint', latest_checkpoint)

    # Override with checkpoint
    config_dict['checkpoint'] = latest_checkpoint

    # Backwards compatibility with new features
    new_features = ['context_input_only', 'emo_weight', 'emotion',
                    'infersent', 'infersent_weight', 'load_rl_ckpt']
    for feat in new_features:
        if feat not in config_dict:
            config_dict[feat] = 0

    # Calc novel embedding metrics
    config_dict['calc_novel_embedding'] = True

    return Config(**config_dict)
