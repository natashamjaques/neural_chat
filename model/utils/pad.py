import torch
from torch.autograd import Variable
from .convert import to_var
from .vocab import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


def pad(tensor, length, dtype=None):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat(
                [var, torch.zeros(
                    length - var.size(0), *var.size()[1:], dtype=dtype).cuda()])
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat(
                [tensor, torch.zeros(
                    length - tensor.size(0), *tensor.size()[1:], dtype=dtype).cuda()])
        else:
            return tensor


def pad_and_pack(tensor_list):
    length_list = ([t.size(0) for t in tensor_list])
    max_len = max(length_list)
    padded = [pad(t, max_len) for t in tensor_list]
    packed = torch.stack(padded, 0)
    return packed, length_list

def pad_tokens(tokens, max_sentence_length=30):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

def pad_conversation(conversation, max_sentence_length=30):
    conversation = [pad_tokens(sentence, 
        max_sentence_length=max_sentence_length) for sentence in conversation]
    return conversation

def pad_sentences(conversations, max_sentence_length=30, max_conversation_length=10):
    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        sentence_length = [min(len(sentence) + 1, max_sentence_length) # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    # [n_conversations, n_sentence (various), max_sentence_length]
    sentences = all_padded_sentences
    # [n_conversations, n_sentence (various)]
    sentence_length = all_sentence_length
    return sentences, sentence_length