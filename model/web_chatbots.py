from collections import OrderedDict
from abc import ABC, abstractmethod
from model.solver import Solver, VariationalSolver
from model.data_loader import get_loader
from model.configs import get_config_from_dir
from model.utils import Vocab, Tokenizer
import os
import numpy as np
import pickle
from model.models import VariationalModels
import emoji

chatbots = OrderedDict()

# Use environment variable for base path
if 'BASE_PATH' not in os.environ:
    base_path = "/home/dialog/checkpoints/"
else:
    base_path = os.environ['BASE_PATH']


def registerbot(botclass):
    bot = botclass()
    chatbots[bot.id] = bot


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Chatbot(ABC):
    def __init__(self, id, name, checkpoint_path, max_conversation_length=5, 
                 max_sentence_length=30, is_test_bot=False, rl=False):
        """
        All chatbots should extend this class and be registered with the @registerbot decorator
        :param id: An id string, must be unique!
        :param name: A user-friendly string shown to the end user to identify the chatbot. Should be unique.
        :param checkpoint_path: Directory where the trained model checkpoint is saved.
        :param max_conversation_length: Maximum number of conversation turns to condition on.
        :param max_sentence_length: Maximum number of tokens per sentence.
        :param is_test_bot: If True, this bot it can be chosen from the list of 
            bots you see at /dialogadmins screen, but will never be randomly 
            assigned to users landing on the home page.
        """
        self.id = id
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.max_conversation_length = max_conversation_length
        self.max_sentence_length = max_sentence_length
        self.is_test_bot = is_test_bot

        self.config = get_config_from_dir(checkpoint_path, mode='test', 
                                          load_rl_ckpt=rl)
        self.config.beam_size = 5

        print('Loading Vocabulary...')
        self.vocab = Vocab()
        self.vocab.load(self.config.word2id_path, self.config.id2word_path)
        print(f'Vocabulary size: {self.vocab.vocab_size}')

        self.config.vocab_size = self.vocab.vocab_size

        # If checkpoint is for an emotion model, load that pickle file
        emotion_sentences = None
        if self.config.emotion:
            emotion_sentences = load_pickle(self.config.emojis_path)

        # Load infersent embeddings if necessary
        infersent_sentences = None
        if self.config.infersent:
            print('Loading infersent sentence embeddings...')
            infersent_sentences = load_pickle(self.config.infersent_path)
            embedding_size = infersent_sentences[0][0].shape[0]
            self.config.infersent_output_size = embedding_size

        self.data_loader = get_loader(
            sentences=load_pickle(self.config.sentences_path),
            conversation_length=load_pickle(self.config.conversation_length_path),
            sentence_length=load_pickle(self.config.sentence_length_path),
            vocab=self.vocab,
            batch_size=self.config.batch_size,
            emojis=emotion_sentences)

        if self.config.model in VariationalModels:
            self.solver = VariationalSolver(self.config, None, self.data_loader, 
                                            vocab=self.vocab, is_train=False)
        else:
            self.solver = Solver(self.config, None, self.data_loader, 
                                 vocab=self.vocab, is_train=False)

        self.solver.build()

    def handle_messages(self, messages):
        """
        Takes a list of messages, and combines those with magic to return a response string
        :param messages: list of strings
        :return: string
        """
        greetings = ["hey , how are you ?", "hi , how 's it going ?", 
                     "hey , what 's up ?", "hi . how are you ?", 
                     "hello , how are you doing today ? ",
                     "hello . how are things with you ?", 
                     "hey ! so, tell me about yourself .",
                     "hi . nice to meet you ."]

        # Check for no response
        if len(messages) == 0:
            # Respond with canned greeting response
            return np.random.choice(greetings)

        # Check for overly short intro messages
        if len(messages) < 2 and len(messages[0]) <= 6: # 6 for "hello."
            first_m = messages[0].lower()
            if 'hi' in first_m or 'hey' in first_m or 'hello' in first_m:
                # Respond with canned greeting response
                return np.random.choice(greetings)

        response = self.solver.generate_response_to_input(
                messages, max_conversation_length=self.max_conversation_length, 
                emojize=True, debug=False)

        return response


""" CORNELL BOTS """

# Cornell baselines
@registerbot
class HRED_cornell(Chatbot):
    def __init__(self):
        super().__init__("HRED_cornell", "HRED movies", base_path + "cornell/hred_canonical")


@registerbot
class VHRED_cornell(Chatbot):
    def __init__(self):
        super().__init__("VHRED_cornell", "VHRED movies", base_path + "cornell/vhred_canonical")


@registerbot
class VHCR_cornell(Chatbot):
    def __init__(self):
        super().__init__("VHCR_cornell", "VHCR movies", base_path + "cornell/vhcr_canonical")


# # Models trained on cornell with deepmoji discriminator
# @registerbot
# class emotion_HRED_cornell(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_HRED_cornell", "ELSA-emo HRED movies", base_path + "cornell/emotion_hred_default")

# @registerbot
# class emotion_VHRED_cornell(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_VHRED_cornell", "pet ELSA", base_path + "cornell/emotion_vhred_default")

# @registerbot
# class emotion_VHRED_cornell_hparam(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_VHRED_cornell_hparam", "ELSA-emo VHRED movies", base_path + "cornell/emotion_vhred_hparam")

# @registerbot
# class emotion_VHCR_cornell(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_VHCR_cornell", "ELSA-emo VCHR movies", base_path + "cornell/emotion_vhcr_default")


# # Models trained on cornell with infersent discriminator
# @registerbot
# class infersent_HRED_cornell(Chatbot):
#     def __init__(self):
#         super().__init__("infersent_HRED_cornell", "ELSA-infer HRED movies", base_path + "cornell/infersent_hred_default")

# @registerbot
# class infersent_VHRED_cornell(Chatbot):
#     def __init__(self):
#         super().__init__("infersent_VHRED_cornell", "ELSA-infer VHRED movies", base_path + "cornell/infersent_vhred_hparam")

# @registerbot
# class infersent_VHCR_cornell(Chatbot):
#     def __init__(self):
#         super().__init__("infersent_VHCR_cornell", "ELSA-infer VCHR movies", base_path + "cornell/infersent_vhcr_default_sentencedrop")


# Models trained on cornell with both discriminators
@registerbot
class emoinfer_HRED_cornell(Chatbot):
    def __init__(self):
        super().__init__("emoinfer_HRED_cornell", "EI HRED movies", base_path + "cornell/emoinfer_hred")


@registerbot
class emoinfer_VHRED_cornell(Chatbot):
    def __init__(self):
        super().__init__("emoinfer_VHRED_cornell", "EI VHRED movies", base_path + "cornell/emoinfer_vhred_hparam")


@registerbot
class emoinfer_VHCR_cornell(Chatbot):
    def __init__(self):
        super().__init__("emoinfer_VHCR_cornell", "EI VCHR movies", base_path + "cornell/emoinfer_vhcr")


""" REDDIT CASUAL BOTS """

# Baseline models trained on reddit_casual
@registerbot
class HRED_reddit_casual(Chatbot):
    def __init__(self):
        super().__init__("HRED_reddit_casual", "HRED reddit",
                         base_path + "reddit_casual/hred_new_hparams",
                         max_conversation_length=10)


@registerbot
class VHRED_reddit_casual(Chatbot):
    def __init__(self):
        super().__init__("VHRED_reddit_casual", "VHRED reddit",
                         base_path + "reddit_casual/vhred_new_hparams",
                         max_conversation_length=10)


@registerbot
class VHCR_reddit_casual(Chatbot):
    def __init__(self):
        super().__init__("VHCR_reddit_casual", "VHCR reddit",
                         base_path + "reddit_casual/vhcr_new_hparams",
                         max_conversation_length=10)


# # Models trained on r/casual with deepmoji discriminator
# @registerbot
# class emotion_HRED_casual(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_HRED_casual", "ELSA-emo HRED reddit", 
#                          base_path + "reddit_casual/emotion_hred_canonical",
#                          max_conversation_length=10)

# @registerbot
# class emotion_VHRED_casual(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_VHRED_casual", "ELSA-emo VHRED reddit", 
#                          base_path + "reddit_casual/emotion_vhred_big_noworddrop", 
#                          max_conversation_length=10)

# @registerbot
# class emotion_VHCR_casual(Chatbot):
#     def __init__(self):
#         super().__init__("emotion_VHCR_casual", "ELSA-emo VHCR reddit", 
#                          base_path + "reddit_casual/emotion_vhcr_hparam",
#                          max_conversation_length=10)


# # Models trained on r/casual with infersent discriminator
# @registerbot
# class infersent_HRED_casual(Chatbot):
#     def __init__(self):
#         super().__init__("infersent_HRED_casual", "ELSA-infer HRED reddit", 
#                          base_path + "reddit_casual/infersent_hred_default",
#                          max_conversation_length=10)

# @registerbot
# class infersent_VHRED_casual(Chatbot):
#     def __init__(self):
#         super().__init__("infersent_VHRED_casual", "ELSA-infer VHRED reddit", 
#                          base_path + "reddit_casual/infersent_vhred_big_noworddrop",
#                          max_conversation_length=10)

# @registerbot
# class infersent_VHCR_casual(Chatbot):
#     def __init__(self):
#         super().__init__("infersent_VHCR_casual", "ELSA-infer VHCR reddit", 
#                          base_path + "reddit_casual/infersent_vhcr_hparam",
#                          max_conversation_length=10)


# Models trained on r/casual with both discriminators
@registerbot
class emoinfer_HRED_casual(Chatbot):
    def __init__(self):
        super().__init__("emoinfer_HRED_casual", "EI HRED reddit", 
                         base_path + "reddit_casual/emoinfer_hred",
                         max_conversation_length=10)


@registerbot
class emoinfer_VHRED_casual(Chatbot):
    def __init__(self):
        super().__init__("emoinfer_VHRED_casual", "EI VHRED reddit", 
                         base_path + "reddit_casual/emoinfer_vhred",
                         max_conversation_length=10)


@registerbot
class emoinfer_VHCR_casual(Chatbot):
    def __init__(self):
        super().__init__("emoinfer_VHCR_casual", "EI VHCR reddit", 
                         base_path + "reddit_casual/emoinfer_vhcr",
                         max_conversation_length=10)
