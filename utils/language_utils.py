import random # creating training, validation, testing set
import re # regexp for replacing easily characters

import unicodedata


def tokenize(list_sentences):
    res_sentences_list = []
    for i in range(len(list_sentences)):
        sentence = list_sentences[i].split(' ')
        while '' in sentence:
            sentence.remove('')
        res_sentences_list.append(sentence)
    return res_sentences_list


def add_PAD_according_to_batch(batch_sentences, pad_symbol):
    # 1. first find the longest sequence here
    batch_size = len(batch_sentences)
    list_of_lengthes = [len(batch_sentences[batch_idx]) for batch_idx in range(batch_size)]
    in_batch_max_seq_len = max(list_of_lengthes)
    batch_num_pads = []
    new_batch_sentences = []
    # 2. add 'PAD' tokens until all the batch have same seq_len
    for batch_idx in range(batch_size):
        num_pads = in_batch_max_seq_len - len(batch_sentences[batch_idx])
        new_batch_sentences.append(batch_sentences[batch_idx] \
            + [pad_symbol] * (num_pads))
        batch_num_pads.append(num_pads)
    return new_batch_sentences, batch_num_pads


def convert_vector_word2idx(sentence, word2idx_dict):
    sentence = [ word2idx_dict[word] for word in sentence]
    return sentence


def convert_allsentences_word2idx(sentences, word2idx_dict):
    res_sentences = []
    for i in range(len(sentences)):
        res_sentences.append(convert_vector_word2idx(sentences[i], word2idx_dict))
    return res_sentences


def convert_vector_idx2word(sentence, idx2word_list):
    sentence = [ idx2word_list[idx] for idx in sentence]
    return sentence


def convert_allsentences_idx2word(sentences, idx2word_list):
    res_sentences = []
    for i in range(len(sentences)):
        res_sentences.append(convert_vector_idx2word(sentences[i], idx2word_list))
    return res_sentences