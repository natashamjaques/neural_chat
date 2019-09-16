"""Library of functions for calculating rewards
Note that rewards should be normalized for best results.
"""
import os
import string
import pickle
from pathlib import Path

import numpy as np
import gensim
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow_hub as hub

from model.utils import embedding_metric, Tokenizer, detokenize
from torchMoji.api.botmoji import Botmoji
from inferSent.api.botsent import Botsent
from Toxicity.toxic import NBLogisticRegression, NBTfidfVectorizer, tokenize


EPSILON = np.finfo(np.float32).eps
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tokenizer = Tokenizer('spacy')
stopwords = stopwords.words('english')
question_words = {'who', 'what', 'why', 'where', 'how', 'when'}
_ = [stopwords.remove(q) for q in question_words]
punct = list(string.punctuation)
contractions = ["'s", "'d", "'ld", "n't", "'re", "'ll", "'ve"]
filters = set(stopwords + contractions + punct)


def _get_emojis():
    # All emojis in the order returned by deepmoji
    EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: :pensive: " + \
             ":ok_hand: :blush: :heart: :smirk: :grin: :notes: :flushed: " + \
             ":100: :sleeping: :relieved: :relaxed: :raised_hands: " + \
             ":two_hearts: :expressionless: :sweat_smile: :pray: " + \
             ":confused: :kissing_heart: :heartbeat: :neutral_face: " + \
             ":information_desk_person: :disappointed: :see_no_evil: " + \
             ":tired_face: :v: :sunglasses: :rage: :thumbsup: :cry: " + \
             ":sleepy: :yum: :triumph: :hand: :mask: :clap: :eyes: :gun: " + \
             ":persevere: :smiling_imp: :sweat: :broken_heart: " + \
             ":yellow_heart: :musical_note: :speak_no_evil: :wink: :skull: " + \
             ":confounded: :smile: :stuck_out_tongue_winking_eye: :angry: " + \
             ":no_good: :muscle: :facepunch: :purple_heart: " + \
             ":sparkling_heart: :blue_heart: :grimacing: :sparkles:"
    EMOJIS = EMOJIS.split(' ')
    return EMOJIS


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
        ':unamused:': -1, ':angry:': -1,  # removing ':hand:': -1 due to ambiguity
        ':rage:': -1
    }
    return emojis_to_rewards


def _get_reward_multiplier():
    EMOJIS = _get_emojis()
    emojis_to_rewards = _get_emojis_to_rewards_dict()
    reward_multiplier = np.zeros(len(EMOJIS))
    for emoji, reward_val in emojis_to_rewards.items():
        loc = EMOJIS.index(emoji)
        reward_multiplier[loc] = reward_val
    return reward_multiplier


def normalizeZ(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + EPSILON)


def discount(rewards, gamma=0.9):
    """Convert raw rewards from batch of episodes to discounted rewards.
    Args:
        rewards: [batch_size, episode_len]
    Returns:
        discounted: [batch_size, episode_len]
    """
    batch_size = rewards.shape[0]
    episode_len = rewards.shape[1]
    discounted = np.zeros_like(rewards)
    running_add = np.zeros((batch_size))
    for step in reversed(range(episode_len)):
        running_add = gamma * running_add + rewards[:, step]
        discounted[:, step] = running_add
    return discounted


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / np.sqrt((np.sum(a * a, axis=1) * np.sum(b * b, axis=1)))


def reward_you(conversations):
    """Allocates reward for any sentence that contains the reward 'you'. Used
    for debugging

    Args:
        conversations: list of lists containing batch_size conversations
        and each conversation is of length 2 * episode_len + 1
    Returns:
        rewards: numpy array of size [batch_size, episode_len]
    """
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened bot responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    rewards = np.array([resp.count('you') for resp in bot_responses])
    rewards = rewards.reshape(num_convs, episode_len)
    return rewards


def reward_conversation_repetition(conversations):
    """Allocates negative reward if the bot repeats a word it has said in a
    previous conversation turn.
    """
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    rewards = np.zeros((num_convs, episode_len))

    for i in range(num_convs):
        conv = conversations[i]
        bot_responses = conv[1::2]
        tokenized = [resp.split() for resp in bot_responses]
        filtered = [set(resp).difference(filters) for resp in tokenized]

        for j in range(1, episode_len):
            current = filtered[j]
            prev = set.union(*filtered[:j])
            repeats = current.intersection(prev)
            rewards[i][j] = len(repeats)

    # You get more negative rewards with more repeats
    return -1 * rewards


def reward_utterance_repetition(conversations):
    """Allocates negative reward if a bot repeats in its current utterance.
    """
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    rewards = np.zeros((num_convs, episode_len))

    for i in range(num_convs):
        conv = conversations[i]
        bot_responses = conv[1::2]
        tokenized = [resp.split() for resp in bot_responses]
        filtered = [[w for w in resp if w not in filters] for resp in tokenized]

        for j in range(episode_len):
            repeats = len(filtered) - len(set(filtered))
            rewards[i][j] = repeats

    # You get more negative rewards with more repeats
    return -1 * rewards


def reward_bot_response_length(conversations):
    """Allocates reward for longer bot outputs/responses.
    """
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened bot responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]

    # Clean punctuation to avoid ? ? ? ? ? long responses
    punct_map = str.maketrans('', '', string.punctuation)
    bot_responses = [resp.translate(punct_map) for resp in bot_responses]
    response_length = [len(resp.split()) for resp in bot_responses]
    rewards = np.array(response_length).reshape(num_convs, episode_len)
    return rewards


# caveats: if the sentiment is negative, it may only be because of the topic, not the person being unhappy with the bot
def reward_user_deepmoji(conversations):
    """Allocates reward based on deepmoji sentiment of user response"""
    # Init deepmoji just once
    if 'botmoji' not in globals():
        print('Loading deepmoji')
        global botmoji
        botmoji = Botmoji()

    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened user responses
    user_responses = [resp for conv in conversations for resp in conv[2::2]]

    # Run deepmoji
    reward_multiplier = _get_reward_multiplier()
    user_emojis = botmoji.encode_multiple(user_responses)
    rewards = np.dot(user_emojis, reward_multiplier)

    for i, resp in enumerate(user_responses):
        if '<unk>' in user_responses[i]:
            rewards[i] = -0.5

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards


def reward_bot_deepmoji(conversations):
    """Allocates reward based on deepmoji sentiment of bot utterance"""
    # Init deepmoji just once
    if 'botmoji' not in globals():
        print('Loading deepmoji')
        global botmoji
        botmoji = Botmoji()

    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened bot responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    for i, conv in enumerate(conversations):
        if len(conv) != 7:
            print(i, conv)
    print(num_convs, episode_len, len(bot_responses))

    # Run deepmoji
    reward_multiplier = _get_reward_multiplier()
    bot_emojis = botmoji.encode_multiple(bot_responses)
    rewards = np.dot(bot_emojis, reward_multiplier)

    for i, resp in enumerate(bot_responses):
        if '<unk>' in bot_responses[i]:
            rewards[i] = -0.5

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards


def reward_deepmoji_coherence(conversations):
    """Allocates reward for coherence between user input and bot response in
    DeepMoji prediction vector space"""
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    user_responses = [resp for conv in conversations for resp in conv[2::2]]

    user_emojis = botmoji.encode_multiple(user_responses)
    bot_emojis = botmoji.encode_multiple(bot_responses)
    emoji_coherence = cosine_similarity(user_emojis, bot_emojis)
    rewards = emoji_coherence.reshape(num_convs, episode_len)
    return rewards


def reward_infersent_coherence(conversations):
    """Allocates reward for coherence between user input and bot response in
    Infersent embedding space"""
    # Init infersent just once
    if 'botsent' not in globals():
        print('Loading InferSent')
        global botsent
        dataset_dir = Path(ROOT_DIR).joinpath('datasets/reddit_casual/train')
        botsent = Botsent(dataset_dir, use_pca=False)

    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    user_responses = [resp for conv in conversations for resp in conv[2::2]]

    user_embed = botsent.encode_multiple(user_responses)
    bot_embed = botsent.encode_multiple(bot_responses)
    coherence = cosine_similarity(user_embed, bot_embed)
    rewards = coherence.reshape(num_convs, episode_len)
    return rewards


def reward_word_similarity(conversations):
    """Allocates reward when bot repeats word appearing in user utterance
    """
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    user_responses = [resp for conv in conversations for resp in conv[2::2]]

    user_tokenized = [sent.split() for sent in user_responses]
    bot_tokenized = [sent.split() for sent in bot_responses]

    # Don't reward for repeating stopwords, question words, or <unknown>
    filter = set.union(filters, question_words, {'<unk>'})
    bot_filtered = [set(resp).difference(filter)
                    for resp in bot_tokenized]

    rewards = np.zeros(num_convs * episode_len)
    for i in range(num_convs * episode_len):
        in_common = [w for w in bot_filtered[i] if w in user_tokenized[i]]

        # Normalize by response len to prevent spamming response
        if len(bot_tokenized[i]):
            rewards[i] = len(in_common) / len(bot_tokenized[i])

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards


def reward_toxicity(conversations):
    """Allocates negative reward if bot response is toxic
    """
    if 'toxicity_model' not in globals():
        print('Loading toxicity model')
        global toxicity_model
        toxic_path = os.path.join(ROOT_DIR, 'Toxicity', 'toxicity_model.pkl')
        toxicity_model = pickle.load(open(toxic_path, 'rb'))

    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    rewards = np.zeros(num_convs * episode_len)

    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    bot_responses = [detokenize(s) for s in bot_responses]

    # Probability that response is toxic
    toxicity = toxicity_model.predict_proba(bot_responses)

    # Hacky way to stop model from talking about depression
    for i, resp in enumerate(bot_responses):
        if 'depressed' in resp:
            toxicity[i] = 0.2

    rewards = toxicity.reshape(num_convs, episode_len)

    # Want to maximize negative toxicity
    return -1 * rewards


def reward_USE_similarity(conversations):
    """Allocates reward for coherence between user input and bot response in
    Universal Sentence Encoder embedding space"""
    if 'universal_encoder' not in globals():
        print('Loading Universal Sentence Encoder')
        global universal_encoder, sess, sents, embed_op
        use_path = os.path.join(ROOT_DIR, "UniversalSentenceEncoder")

        with tf.device('/cpu:0'):
            universal_encoder = hub.Module(use_path)
            sents = tf.placeholder(tf.string, shape=None, name="input_sents")
            embed_op = universal_encoder(sents)

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    user_responses = [resp for conv in conversations for resp in conv[2::2]]

    user_embed = sess.run(embed_op, feed_dict={sents: user_responses})
    bot_embed = sess.run(embed_op, feed_dict={sents: bot_responses})
    similarity = cosine_similarity(user_embed, bot_embed)
    rewards = similarity.reshape(num_convs, episode_len)

    return rewards


def reward_question(conversations):
    """Allocates reward for any bot utterance that asks questions."""
    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    rewards = np.zeros(num_convs * episode_len)

    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    question_words = ['who', 'what', 'why', 'where', 'how', 'when']

    for i, resp in enumerate(bot_responses):
        resp = resp.lower()
        if any(q in resp for q in question_words) and '?' in resp:
            rewards[i] += 1

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards


def reward_word2vec_coherence(conversations):
    """Allocates reward for coherence between user input and bot response in
    word2vec embedding space"""
    if 'word2vec' not in globals():
        print('Loading word2vec dict')
        global word2vec, keys
        word2vec_path = os.path.join(ROOT_DIR, "datasets/GoogleNews-vectors-negative300.bin")
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        keys = word2vec.vocab

    num_convs = len(conversations)
    episode_len = (len(conversations[0]) - 1) // 2
    # Flattened responses
    bot_responses = [resp for conv in conversations for resp in conv[1::2]]
    user_responses = [resp for conv in conversations for resp in conv[2::2]]

    user_inputs = [sent.split() for sent in user_responses]
    bot_responses = [sent.split() for sent in bot_responses]

    user_inputs_w2v = [[word2vec[w] for w in sent if w in keys] for sent in user_inputs]
    bot_responses_w2v = [[word2vec[w] for w in sent if w in keys] for sent in bot_responses]

    rewards = np.zeros(num_convs * episode_len)

    for i in range(num_convs * episode_len):
        if user_inputs_w2v[i] == [] or bot_responses_w2v[i] == []:
            rewards[i] = 0
        else:
            rewards[i] = embedding_metric(
                [user_inputs_w2v[i]], [bot_responses_w2v[i]], word2vec,
                method='average')[0]

    rewards = rewards.reshape(num_convs, episode_len)
    return rewards
