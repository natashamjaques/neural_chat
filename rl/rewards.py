"""Library of functions for calculating rewards on a CsvReplayBuffer object.

Each reward funtion should be named 'reward_<X>', and add a column to the buffer
labeled 'reward_<X>'.

Rewards can modify the buffer by removing rows for which the reward cannot be 
computed.

Note that rewards should be normalized for best results.
"""

from torchMoji.api.botmoji import Botmoji, EMOJIS
from inferSent.api.botsent import Botsent
from model.utils import embedding_metric
import replay_buffer

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

import gensim
import nltk
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
word2vec_path = os.path.join(os.path.join(ROOT_DIR, 'datasets'), 'GoogleNews-vectors-negative300.bin')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experience_path', type=str, default=None,
                        help='Path to a .csv containing experiences')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save csv with computed reward')
    parser.add_argument('--raw', action='store_true',
                        help='Set to True if processing raw file from website')
    parser.set_defaults(raw=False)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--infersent_data', type=str, default='cornell',
                        help='Dataset to use for infersent model')
    parser.add_argument('--study_key', type=str, default=None,
                        help='Limit analysis to a particular study')
    return parser.parse_args()


def _get_emojis_to_rewards_dict():
    # How detected emojis map to rewards
    emojis_to_rewards = {
        # very strongly positive
        ':kissing_heart:': 1, ':thumbsup:': 1, ':ok_hand:': 1,
        ':smile:': 1,

        # strongly positive
        ':blush:': 0.75, ':wink:': 0.75, ':muscle:': 0.75,
        ':grin:': 0.75, ':heart_eyes:': 0.75, ':100:': 0.75,

        # positive
        ':smirk:': 0.5, ':stuck_out_tongue_winking_eye:': 0.5,
        ':sunglasses:': 0.5, ':relieved:': 0.5, ':relaxed:': 0.5,
        ':blue_heart:': 0.5, ':two_hearts:': 0.5, ':heartbeat:': 0.5,
        ':yellow_heart:': 0.5,

        # negative
        ':disappointed:': -0.5, ':eyes:': -0.5,
        ':expressionless:': -0.5, ':sleeping:': -0.5,
        ':grimacing:': -0.5,

        # strongly negative
        ':neutral_face:': -0.75, ':confused:': -0.75,
        ':triumph:': -0.75, ':confounded:': -0.75,

        # very strongly negative
        ':unamused:': -1, ':angry:': -1,
        ':rage:': -1
    }
    return emojis_to_rewards


def _get_reward_multiplier():
    emojis_to_rewards = _get_emojis_to_rewards_dict()
    reward_multiplier = np.zeros(len(EMOJIS))
    for emoji, reward_val in emojis_to_rewards.items():
        loc = EMOJIS.index(emoji)
        reward_multiplier[loc] = reward_val
    return reward_multiplier


def normalize_01(x):
    x = np.array(x)
    min_x = min(x)
    max_x = max(x)
    return (x - min_x) / (max_x - min_x)


def normalize_z(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward
    From: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / np.sqrt(
        (np.sum(a * a, axis=1) * np.sum(b * b, axis=1)))


def reward_you(csv_buff):
    """Allocates reward for any sentence that contains the reward 'you'. 
    Simple reward used for debugging"""
    print("Computing reward_you")
    rewards = [1 if 'you' in r else 0 for r in csv_buff.buffer['Response']]
    csv_buff.buffer['reward_you'] = normalize_01(rewards)
    return csv_buff


def reward_answer_length_words(csv_buff):
    """Allocates reward for number of words in the human answer."""
    print("Computing reward_answer_length_words")
    dones = csv_buff.buffer['done'].tolist()
    user_responses = [x[-1] for x in csv_buff.buffer['next_state'].tolist()]
    rewards = [0] * len(user_responses)
    for i, x in enumerate(user_responses):
        if not dones[i]:
            rewards[i] = len(x.split(' '))
    csv_buff.buffer['reward_answer_length_words_unnormalized'] = rewards

    # Clip to a max length of 30 words to prevent noise from spammers
    rewards = np.clip(rewards, 0, 30)
    csv_buff.buffer['reward_answer_length_words'] = normalize_01(rewards)
    csv_buff.buffer['reward_answer_length_words_zscore'] = normalize_z(rewards)
    return csv_buff


def reward_answer_length_chars(csv_buff):
    """Allocates reward for number of characters in the human answer."""
    print("Computing reward_answer_length_chars")
    dones = csv_buff.buffer['done'].tolist()
    user_responses = [x[-1] for x in csv_buff.buffer['next_state'].tolist()]
    rewards = [0] * len(user_responses)
    for i, x in enumerate(user_responses):
        if not dones[i]:
            rewards[i] = len(x)
    csv_buff.buffer['reward_answer_length_chars_unnormalized'] = rewards
    csv_buff.buffer['reward_answer_length_chars'] = normalize_01(rewards)
    return csv_buff


def reward_user_ha(csv_buff):
    """Allocates reward for number 'ha's in the human answer."""
    print("Computing reward_user_ha")
    dones = csv_buff.buffer['done'].tolist()
    user_responses = [x[-1] for x in csv_buff.buffer['next_state'].tolist()]
    rewards = [0] * len(user_responses)
    for i, x in enumerate(user_responses):
        if not dones[i]:
            rewards[i] = x.count('ha')
    rewards = np.clip(rewards, 0, 4)
    csv_buff.buffer['reward_user_ha'] = normalize_z(rewards)
    return csv_buff


def reward_conversation_length(csv_buff, gamma=.95):
    """Allocates reward for length of each conversation"""
    print("Computing reward_conversation_length")
    df = csv_buff.buffer
    processed_df = None
    for chat_id in df['Chat ID'].unique():
        conv = df[df['Chat ID'] == chat_id]
        conv = conv.sort_values('Datetime')

        rewards = [0.0] * len(conv)
        # Leave last element of convo as 0 because user chose to leave at this.
        if len(conv) > 1:
            rewards[-2] = float(len(conv))
            rewards = discount_rewards(rewards, gamma)

        conv['reward_conversation_length'] = rewards
        conv['reward_conversation_length_unnormalized'] = [len(conv)] * len(conv)
        if processed_df is None:
            processed_df = conv
        else:
            processed_df = pd.concat([processed_df, conv])
    processed_df['reward_conversation_length'] = normalize_z(
        processed_df['reward_conversation_length'].tolist())
    csv_buff.buffer = processed_df
    return csv_buff


def reward_manual_ratings(csv_buff):
    """Allocates reward for manual presses of upvote/downvote button"""
    print("Computing reward_manual_ratings")
    rewards = csv_buff.buffer['Response Rating'].tolist()
    csv_buff.buffer['reward_manual_ratings'] = normalize_z(rewards)
    return csv_buff


# caveats: if the sentiment is negative, it may only be because of the topic, 
# not the person being unhappy with the bot
def reward_deepmoji(csv_buff):
    """Allocates reward based on deepmoji sentiment of user response"""
    print("Computing reward_deepmoji")

    # Create reward values
    reward_multiplier = _get_reward_multiplier()

    # Run deepmoji on user responses
    botmoji = Botmoji()
    user_responses = [x[-1] for x in csv_buff.buffer['next_state'].tolist()]
    mask = [0 if response == '' else 1 for response in user_responses]
    user_emojis = botmoji.encode_multiple(user_responses)
    rewards = np.dot(user_emojis, reward_multiplier)
    rewards *= mask
    csv_buff.buffer['reduced_deepmoji'] = rewards
    csv_buff.buffer['reward_deepmoji'] = normalize_z(rewards)
    return csv_buff, botmoji


def reward_deepmoji_coherence(csv_buff, botmoji=None):
    """Allocates reward for coherence between user input and bot response in 
    DeepMoji prediction vector space"""
    print("Computing reward_deepmoji_coherence")
    if not botmoji:
        botmoji = Botmoji()

    user_queries = [x[-1] for x in csv_buff.buffer['state'].tolist()]
    bot_answers = csv_buff.buffer['action'].tolist()

    assert_msg = "Different number of user queries and bot answers"
    assert (len(bot_answers) == len(user_queries)), assert_msg

    user_emojis = botmoji.encode_multiple(user_queries)
    bot_emojis = botmoji.encode_multiple(bot_answers)
    emoji_coherence = cosine_similarity(user_emojis, bot_emojis)
    csv_buff.buffer['reward_deepmoji_coherence_unnormalized'] = emoji_coherence
    csv_buff.buffer['reward_deepmoji_coherence'] = emoji_coherence - 0.5
    return csv_buff


def reward_infersent_coherence(csv_buff, dataset):
    """Allocates reward for coherence between user input and bot response in 
    Infersent embedding space"""
    print("Computing reward_infersent_coherence")
    repo_dir = Path(ROOT_DIR)
    dataset_dir = repo_dir.joinpath('datasets', dataset, 'train')
    botsent = Botsent(dataset_dir, use_pca=False)

    user_queries = [x[-1] for x in csv_buff.buffer['state'].tolist()]
    bot_answers = csv_buff.buffer['action'].tolist()

    assert_msg = "Different number of user queries and bot answers"
    assert (len(bot_answers) == len(user_queries)), assert_msg

    user_embed = botsent.encode_multiple(user_queries)
    bot_embed = botsent.encode_multiple(bot_answers)
    coherence = cosine_similarity(user_embed, bot_embed)

    reward_name = 'reward_infersent_coherence_' + dataset
    csv_buff.buffer[reward_name+'_unnormalized'] = coherence
    csv_buff.buffer[reward_name] = normalize_z(coherence)

    return csv_buff


def reward_user_emotional_transition(csv_buff, botmoji=None):
    """Allocates reward for improvement in the sentiment of the user
    on a turn-by-turn basis, by calculating a weighted sum over the change of 
    deepmoji softmax values."""
    print("Computing reward_user_emotional_transition")

    reward_multiplier = _get_reward_multiplier()
    if not botmoji:
        botmoji = Botmoji()

    user_queries = [x[-1] for x in csv_buff.buffer['state'].tolist()]
    future_user_queries = [x[-1] for x in csv_buff.buffer['next_state'].tolist()]

    assert_msg = "Different number of user queries and future user queries"
    assert (len(future_user_queries) == len(user_queries)), assert_msg

    user_emojis = botmoji.encode_multiple(user_queries)
    future_user_emojis = botmoji.encode_multiple(future_user_queries)
    diff_emojis = future_user_emojis - user_emojis
    emoji_transition = np.dot(diff_emojis, reward_multiplier)

    csv_buff.buffer['reward_user_emotional_transition'] = emoji_transition
    return csv_buff


def reward_user_min_max_emotion_transition(csv_buff, botmoji=None):
    """Allocates reward for change in emotional tone of user response between 
    min and max points of conversation."""
    print("Computing reward_user_min_max_emotion_transition")

    if 'reduced_deepmoji' not in csv_buff.buffer.columns:
        csv_buff, botmoji = reward_deepmoji(csv_buff, botmoji)

    df = csv_buff.buffer
    processed_df = None
    for chat_id in df['Chat ID'].unique():
        conv = df[df['Chat ID'] == chat_id]
        conv = conv.sort_values('Datetime').reset_index(drop=False)
        conv_sorted_deepmoji = conv.sort_values('reduced_deepmoji').reset_index(
            drop=True)
        
        min_emo = conv_sorted_deepmoji['reduced_deepmoji'].iloc[0]
        max_emo = conv_sorted_deepmoji['reduced_deepmoji'].iloc[-1]
        min_idx = conv_sorted_deepmoji['index'].iloc[0]
        max_idx = conv_sorted_deepmoji['index'].iloc[-1]
        min_max_transition = (min_emo - max_emo) / (min_idx - max_idx)
        if not np.isfinite(min_max_transition):
            rewards = [0] * len(conv)
        else:
            rewards = [min_max_transition] * len(conv)
        conv['reward_user_min_max_emotion_transition'] = rewards
        if processed_df is None:
            processed_df = conv
        else:
            processed_df = pd.concat([processed_df, conv])
    csv_buff.buffer = processed_df
    return csv_buff


def reward_user_var_emotion(csv_buff, botmoji=None):
    """Allocates reward for variability in emotional tone of user response."""
    print("Computing reward_user_var_emotion")

    if 'reduced_deepmoji' not in csv_buff.buffer.columns:
        csv_buff, botmoji = reward_deepmoji(csv_buff, botmoji)

    df = csv_buff.buffer
    processed_df = None
    for chat_id in df['Chat ID'].unique():
        conv = df[df['Chat ID'] == chat_id]
        rewards = [np.var(conv['reduced_deepmoji'])] * len(conv)
        conv['reward_user_var_emotion'] = rewards
        if processed_df is None:
            processed_df = conv
        else:
            processed_df = pd.concat([processed_df, conv])
    csv_buff.buffer = processed_df
    return csv_buff


def reward_user_auc_emotion_transition(csv_buff, botmoji=None):
    """Allocates reward for positive AUC for shift in emotional tone of user 
    responses."""
    print("Computing reward_user_auc_emotion_transition")

    if 'reward_user_emotional_transition' not in csv_buff.buffer.columns:
        csv_buff = reward_user_emotional_transition(csv_buff, botmoji)

    df = csv_buff.buffer
    processed_df = None
    for chat_id in df['Chat ID'].unique():
        conv = df[df['Chat ID'] == chat_id]
        rewards = [np.sum(conv['reward_user_emotional_transition'])] * len(conv)
        conv['reward_user_auc_emotion_transition'] = rewards
        if processed_df is None:
            processed_df = conv
        else:
            processed_df = pd.concat([processed_df, conv])
    csv_buff.buffer = processed_df
    return csv_buff


def reward_question(csv_buff):
    """Allocates reward for any sentence that contains asks questions."""
    print("Computing reward_question")

    bot_answers = csv_buff.buffer['action'].tolist()

    question_words = ['who', 'what', 'why', 'where', 'how', 'when']
    rewards = [0.5 if '?' in r else 0 for r in bot_answers]
    for i, bot_ans in enumerate(bot_answers):
        has_q_word = False
        for q_word in question_words:
            if q_word in nltk.word_tokenize(bot_ans.lower()):
                has_q_word = True
                break
        if has_q_word:
            rewards[i] += 0.5
    csv_buff.buffer['reward_question'] = rewards
    csv_buff.buffer['reward_question_zscore'] = normalize_z(rewards)
    return csv_buff


def reward_traditional_embedding_metrics(csv_buff):
    """Allocates reward for any sentence that contains asks questions."""
    print("Computing reward_traditional_embedding_metrics")

    print('Tokenizing')
    user_queries = [x[-1] for x in csv_buff.buffer['state'].tolist()]
    bot_answers = csv_buff.buffer['action'].tolist()

    user_queries = [nltk.word_tokenize(sent) for sent in user_queries]
    bot_answers = [nltk.word_tokenize(sent) for sent in bot_answers]

    print('Loading word2vec model')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, 
                                                               binary=True)
    keys = word2vec.vocab

    print('Calculating distances')
    user_q_w2v = [[word2vec[w] for w in sent if w in keys] for sent in user_queries]
    bot_ans_w2v = [[word2vec[w] for w in sent if w in keys] for sent in bot_answers]

    rewards_average = np.zeros(len(user_queries))
    rewards_extrema = np.zeros(len(user_queries))
    rewards_greedy = np.zeros(len(user_queries))

    for i in range(len(user_queries)):
        if bot_ans_w2v[i] == [] or user_q_w2v[i] == []:
            rewards_average[i] = 0
            rewards_extrema[i] = 0
            rewards_greedy[i] = 0
        else:
            rewards_average[i] = embedding_metric(
                [bot_ans_w2v[i]], [user_q_w2v[i]], word2vec, 
                method='average')[0]
            rewards_extrema[i] = embedding_metric(
                [bot_ans_w2v[i]], [user_q_w2v[i]], word2vec, 
                method='extrema')[0]
            rewards_greedy[i] = embedding_metric(
                [bot_ans_w2v[i]], [user_q_w2v[i]], word2vec, method='greedy')[0]

    csv_buff.buffer['rewards_embedding_average'] = rewards_average
    csv_buff.buffer['rewards_embedding_extrema'] = rewards_extrema
    csv_buff.buffer['rewards_embedding_greedy'] = rewards_greedy
    return csv_buff


def reward_compliments(csv_buff):
    key_phrases = ['you are beautiful', "you are so beautiful",     
                    "you're beautiful", "you re beautiful", 
                    'you are the best', "you're the best", 
                   'i like you', "you're a good", 
                   'you re a good', 'i love the way you']
    bot_answers = csv_buff.buffer['action'].tolist()
    
    rewards = [0] * len(bot_answers)
    good_answers = []
    for i, b in enumerate(bot_answers):
        for k in key_phrases:
            if k in b:
                good_answers.append(b)
                rewards[i] = 1
                continue

    csv_buff.buffer['reward_compliments'] = rewards
    csv_buff.buffer['reward_compliments_zscore'] = normalize_z(rewards)
    return csv_buff


def reward_politeness(csv_buff):
    key_phrases = ['if i may', 'may i', 'please', 'thanks', 'no worries',
                   "if you don't mind", "have a great day", "i'm sorry"]
    bot_answers = csv_buff.buffer['action'].tolist()
    # look in bot answers for "mind"
    mind = [m for m in bot_answers if "mind" in m]
    
    rewards = [0] * len(bot_answers)
    for i, b in enumerate(bot_answers):
        for k in key_phrases:
            if k in b:
                rewards[i] = 1
                continue

    csv_buff.buffer['reward_politeness'] = rewards
    csv_buff.buffer['reward_politeness_zscore'] = normalize_z(rewards)
    return csv_buff


def reward_supportive(csv_buff):
    key_phrases = ["you're right", "you are right", "you re right", 
                   "you're not alone", "you are not alone", "you re not alone", 
                   "congrats", "that's a good idea", "that is a good idea", 
                   "you'll be fine", "you will be fine", "you ll be fine", 
                   "you'll be okay", "you will be okay", "you ll be okay",
                   "it will get better", "sorry you're going through", 
                   "sorry you are going through this", 
                   "sorry you re going through this", 
                   "if it makes you feel better", 
                   "if it makes you feel any better", "keep your head up", 
                   "keep it up", "i'm in a similar situation", 
                   "i am in a similar situation", "i m in a similar situation", 
                   "you'll get it", "you will get it", "you ll get it", 
                   "happy for you", "i'm in the same boat", 
                   "i am in the same boat", "if you feel like you need to vent"]
    bot_answers = csv_buff.buffer['action'].tolist()
    rewards = [0] * len(bot_answers)
    for i, b in enumerate(bot_answers):
        for k in key_phrases:
            if k in b:
                rewards[i] = 1
                continue
    csv_buff.buffer['reward_supportive'] = rewards
    csv_buff.buffer['reward_supportive_zscore'] = normalize_z(rewards)
    return csv_buff


def reward_cheerful(csv_buff):
    key_phrases = ["nice to hear", "happy", "excited", "really nice", 
                   "glad", "the best", "great", "good time", "looking forward", 
                   "beautiful"]
    bot_answers = csv_buff.buffer['action'].tolist()
    rewards = [0] * len(bot_answers)
    for i, b in enumerate(bot_answers):
        for k in key_phrases:
            if k in b:
                rewards[i] = 1
                continue

    csv_buff.buffer['reward_cheerful'] = rewards
    csv_buff.buffer['reward_cheerful_zscore'] = normalize_z(rewards)
    return csv_buff


if __name__ == '__main__':
    """Load experience replay buffer from file and compute rewards on it"""
    kwargs = parse_args()

    buffer = replay_buffer.CsvReplayBuffer(kwargs.experience_path, 
                                           raw=kwargs.raw,
                                           study_key=kwargs.study_key)
    print("\nBuffer contains", len(buffer.buffer), "responses.")

    buffer = reward_you(buffer)
    buffer = reward_answer_length_words(buffer)
    buffer = reward_answer_length_chars(buffer)
    buffer = reward_conversation_length(buffer, kwargs.discount)
    buffer = reward_manual_ratings(buffer)
    buffer, botmoji = reward_deepmoji(buffer)
    buffer = reward_deepmoji_coherence(buffer, botmoji)
    buffer = reward_user_emotional_transition(buffer, botmoji)
    buffer = reward_user_min_max_emotion_transition(buffer, botmoji)
    buffer = reward_user_var_emotion(buffer, botmoji)
    buffer = reward_user_auc_emotion_transition(buffer, botmoji)
    buffer = reward_infersent_coherence(buffer, 'cornell')
    buffer = reward_infersent_coherence(buffer, 'reddit_casual')
    buffer = reward_question(buffer)
    buffer = reward_user_ha(buffer)
    buffer = reward_traditional_embedding_metrics(buffer)
    buffer = reward_compliments(buffer)
    buffer = reward_politeness(buffer)
    buffer = reward_supportive(buffer)
    buffer = reward_cheerful(buffer)

    # Save to processed csv file
    save_path = kwargs.save_path
    if kwargs.study_key is not None:
        save_path = save_path[:-4] + '_' + kwargs.study_key + '.csv'
    buffer.buffer.to_csv(save_path)
