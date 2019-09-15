import argparse
from model.configs import get_config_from_dir
from hrl_tune import REINFORCETuner


def parse_config_args():
    parser = argparse.ArgumentParser()

    # Must provide checkpoint of pre-trained model to load
    parser.add_argument('--checkpoint', type=str, default=None)

    # RL rewards
    # # use: python run_rl.py -r 'reward_you' 'reward_what' -rw 2.0 1.5
    parser.add_argument('-r', '--rewards', nargs='+',
                        help='<Required> List of reward functions to combine')
    parser.add_argument('-w', '--reward_weights', nargs='+',
                        help='<Required> List of weights on reward functions')
    parser.add_argument('--gamma', type=float, default=0.9)


    # Training config
    parser.add_argument('--rl_mode', type=str, default='train',
                        help='Set to interact to interact with the bot.')
    parser.add_argument('--reinforce', action='store_true',
                        help='Set to True for only word level flat reinforce RL.')
    parser.add_argument('--vhrl', action='store_true',
                        help='Joint training of worker and manager actions.')
    parser.add_argument('--decoupled_vhrl', action='store_true',
                        help='Similar to vhrl but alternate between update worker and manager.')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='weight used to adjust magnitude of manager actions and rewards')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight used to adjust magnitude of worker actions and rewards')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='Number of update steps')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate used for updates if --reinforce or --vhrl')
    parser.add_argument('--manager_learning_rate', type=float, default=0.0001,
                        help='Learning rate used to update manager if --decoupled_vhrl')
    parser.add_argument('--worker_learning_rate', type=float, default=0.000001,
                        help='Learning rate used to update worker if --decoupled_vhrl')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Turn off drop out to reduce training variance')

    # RL simulation config
    # NOTE: Simulated conversation length is 2 * (episode_len) + 1 since we
    # start and end with user/simulator utterance
    parser.add_argument('--episode_len', type=int, default=3,
                        help='Number of bot responses/action in simulated conversation')
    parser.add_argument('--rl_batch_size', type=int, default=35,
                        help='Number of conversations simulated at each train step')
    parser.add_argument('--beam_size', type=int, default=5)

    # Conversation config
    parser.add_argument('-s', '--max_sentence_length', type=int, default=30)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=5)

    # Loading previously trained RL models
    parser.add_argument('--load_rl_ckpt', action='store_true',
                        help='Indicates that an RL checkpoint should be loaded')
    parser.add_argument('--rl_ckpt_epoch', type=int, default=None)

    # Logging configs
    parser.add_argument('--print_every_n', type=int, default=25,
                        help='print training statistics evey n steps')
    parser.add_argument('--log_every_n', type=int, default=25,
                        help='save trainings statistics evey n steps')
    parser.add_argument('--save_every_n', type=int, default=200,
                        help='save model evey n steps')
    parser.add_argument('--extra_save_dir', type=str, default='')

    return vars(parser.parse_args())


if __name__ == '__main__':

    kwargs_dict = parse_config_args()

    # Default rewards if user is too lazy to provide
    if not kwargs_dict['rewards']:
        kwargs_dict['rewards'] = ['reward_bot_deepmoji']
    if not kwargs_dict['reward_weights']:
        kwargs_dict['reward_weights'] = [1.0] * len(kwargs_dict['rewards'])

    # Train config
    kwargs_dict['mode'] = 'train'
    config = get_config_from_dir(kwargs_dict['checkpoint'], **kwargs_dict)

    # Val config
    kwargs_dict['mode'] = 'valid'
    val_config = get_config_from_dir(kwargs_dict['checkpoint'], **kwargs_dict)

    tuner = REINFORCETuner(config, val_config)

    if config.rl_mode == 'train':
        tuner.reinforce_learn()
    elif config.rl_mode == 'interact':
        tuner.interact()
    else:
        print("Error, can't understand mode", config.mode)
