#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
baseline.py : establish baseline model
Guoqin Ma <sebsk@stanford.edu>
"""

import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn.utils.rnn import pack_padded_sequence
import sys
import pickle
from vocab import VocabEntry
import numpy as np


class BaselineModel(nn.Module):

    def __init__(self, hidden_size, embedding, vocab, n_class, dropout_rate=0):
        """
        @param hidden_size (int): lstm hidden size
        @param embedding (torch.Tensor): shape (vocab_size, embed_size), glove embedding matrix
        @param vocab (VocabEntry): constructed from glove word2id
        @param n_class (int): number of labels / classes
        @param dropout_rate (float): dropout rate for training
        """

        super(BaselineModel, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embedding.size(1)
        self.vocab_size = embedding.size(0)
        self.vocab = vocab
        self.n_class = n_class
        self.padding_idx = self.vocab['<pad>']
        self.dropout_rate = dropout_rate
        self.device = embedding.device

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.padding_idx)
        self.embedding_layer.weight = nn.Parameter(embedding)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.hidden_size*2, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, sents):
        """
        @param sents (list[list[str]]): sents in word, sorted in descending length

        @return pre_softmax (torch.Tensor): logits to put into softmax function to calculate prob
        """
        text_lengths = torch.tensor([len(sent) for sent in sents], device=self.device)
        sents_tensor = self.vocab.to_input_tensor(sents, device=self.device)  # (max_sent_length, batch_size)
        x_embed = self.embedding_layer(sents_tensor)  # (max_sent_length, batch_size, embed_size)
        seq = pack_padded_sequence(x_embed.float(), text_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(seq)
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)  # (batch_size, n_class)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model

        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BaselineModel(vocab=params['vocab'], embedding=params['embedding'].to(device), **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         n_class=self.n_class),
            'vocab': self.vocab,
            'embedding': self.embedding_layer.weight,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

if __name__ == '__main__':

    glove_word2id = pickle.load(open('glove_word2id', 'rb'))
    glove_word2id.update({'<unk>': len(glove_word2id)})
    glove_word2id.update({'<pad>': len(glove_word2id)})
    vocab = VocabEntry(glove_word2id)

    embedding_matrix = np.load(open('glove_embeddings.npy', 'rb'))
    embedding_matrix = np.vstack((embedding_matrix,
                                  np.random.uniform(-0.1, 0.1, (2, embedding_matrix.shape[1]))))
    glove_embeddings = torch.tensor(embedding_matrix)
    model = BaselineModel(hidden_size=256, embedding=glove_embeddings, vocab=vocab, n_class=7)

    for param in model.parameters():
        print(param)

    print(model.state_dict())