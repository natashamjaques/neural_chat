import argparse
from model.configs import get_config_from_dir
from way_off_policy_batch_rl import BatchQ


def parse_config_args():
    parser = argparse.ArgumentParser()

    # Must provide checkpoint of pre-trained model to load
    parser.add_argument('--checkpoint', type=str, default=None)

    # Info about where batch data is located and whether it needs preprocessing
    parser.add_argument('--experience_path', type=str, default=None,
                        help='Path to .csv containing batch data / experience')
    parser.add_argument('--raw_buffer', action='store_true',
                        help='Set to True if processing buffer from raw file'
                             'obtained directly from website')
    parser.set_defaults(raw=False)

    # RL rewards
    # e.g: python run_rl.py -r 'reward_you' 'reward_what' -rw 2.0 1.5
    parser.add_argument('-r','--rewards', nargs='+',
                        help='<Required> List of reward functions to combine', )
    parser.add_argument('-w','--reward_weights', nargs='+',
                        help='<Required> List of weights on reward functions', )

    # RL config
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--rl_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--q_loss_func', type=str, default="smooth_l1_loss",
                        help='Name of torch loss function, eg. mse_loss')
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # KL-control params
    parser.add_argument('--kl_control', action='store_true',
                        help='Set to True if minimizing KL from prior.')
    parser.set_defaults(kl_control=False)
    parser.add_argument('--kl_weight_c', type=float, default=0.5)
    parser.add_argument('--kl_calc', type=str, default='sample',
                        help="Can be 'integral' for normal KL, or 'sample' to"
                             "just use the logp(a|s) - logq(a|s)")
    parser.add_argument('--psi_learning', action='store_true',
                        help='Set to True if using Psi learning (logsumexp)')
    parser.set_defaults(psi_learning=False)

    # Model averaging
    parser.add_argument('--model_averaging', action='store_true',
                        help='Set to True if minimizing KL from averaged probs')
    parser.set_defaults(model_averaging=False)
    parser.add_argument('--separate_datasets', action='store_true',
                        help="If true, don't merge probabilities of models " + \
                             " from different datasets")
    parser.set_defaults(separate_datasets=False)

    # Uses monte carlo estimates to alleviate optimism in target Q values
    parser.add_argument('--monte_carlo_count', type=int, default=1)

    # Training and logging configs
    parser.add_argument('--log_every_n', type=int, default=20)
    parser.add_argument('--save_every_n', type=int, default=200)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--extra_save_dir', type=str, default='')
    parser.add_argument('--rl_mode', type=str, default='train',
                        help='Set to interact to interact with the bot.')
    parser.add_argument('--beam_size', type=int, default=5)

    # Conversation args
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=5)

    # Loading previously trained RL models
    parser.add_argument('--load_rl_ckpt', action='store_true',
                        help='Indicates that an RL checkpoint should be loaded')
    parser.set_defaults(load_rl_ckpt=False)
    parser.add_argument('--rl_ckpt_epoch', type=int, default=None)

    return vars(parser.parse_args())


if __name__ == '__main__':

    kwargs_dict = parse_config_args()

    # Default rewards if user is too lazy to provide
    if not kwargs_dict['rewards']:
        kwargs_dict['rewards'] = ['reward_conversation_length']
    if not kwargs_dict['reward_weights']:
        kwargs_dict['reward_weights'] = [1.0] * len(kwargs_dict['rewards'])

    # Only one param necessary to invoke model averaging
    if kwargs_dict['model_averaging']:
        kwargs_dict['kl_control'] = True
        kwargs_dict['kl_calc'] = 'sample'

    if kwargs_dict['rl_mode'] == 'interact':
        kwargs_dict['beam_size'] = 5

    # Train config
    kwargs_dict['mode'] = 'train'
    config = get_config_from_dir(kwargs_dict['checkpoint'], **kwargs_dict)

    # Val config
    kwargs_dict['mode'] = 'valid'
    val_config = get_config_from_dir(kwargs_dict['checkpoint'], **kwargs_dict)

    bqt = BatchQ(config, val_config=val_config)

    if config.rl_mode == 'train':
        bqt.q_learn()
    elif config.rl_mode == 'interact':
        bqt.interact()
    else:
        print("Error, can't understand mode", config.mode)
