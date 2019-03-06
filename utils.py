#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
utils.py: modified from Homework 5
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()` 
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal 
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21 

    sents_padded = []
    max_sentence_length = len(max(sents, key=len))
    pad_word = [char_pad_token]*max_word_length
    for sent in sents:
        sent_padded = []
        for word in sent:
            word_len = len(word)
            if word_len > max_word_length:
                sent_padded.append(word[:max_word_length])
            elif word_len <= max_word_length:
                sent_padded.append(word + [char_pad_token]*(max_word_length-word_len))
        sent_padded = sent_padded + [pad_word]*(max_sentence_length-len(sent))
        sents_padded.append(sent_padded)

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded


def read_corpus(file_path):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False, bert=None):
    """ Yield batches of sentences and labels reverse sorted by length (largest to smallest).
    @param data (dataframe): dataframe with ProcessedText (str) and label (int) columns
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    @param bert (str): whether for BERT training. Values: "large", "base", None
    """
    batch_num = math.ceil(data.shape[0] / batch_size)
    index_array = list(range(data.shape[0]))

    if shuffle:
        data = data.sample(frac=1)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]

        if bert:
            examples = data.iloc[indices].sort_values(by='ProcessedText_BERT'+bert+'_length', ascending=False)
            sents = list(examples.ProcessedText_BERT)
        else:
            examples = data.iloc[indices].sort_values(by='ProcessedText_length', ascending=False)
            sents = [text.split(' ') for text in examples.ProcessedText]

        targets = list(examples.InformationType_label.values)
        yield sents, targets  # list[list[str]] if not bert else list[str], list[int]
