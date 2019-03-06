#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
glove_embedding.py : create word2id (dict) and glove embeddings (numpy array) files.
Guoqin Ma <sebsk@stanford.edu>
"""

import os
import pickle
import numpy as np

os.chdir('/Users/sebastianma/Documents/CS221AI/Project/glove.twitter.27B')

with open('glove.twitter.27B.200d.txt') as f:
    words = f.readlines()

word2id = dict()
embeddings = []

for i, w in enumerate(words):
    w_list = w.split(' ')
    word2id.update({w_list[0]: i})
    embeddings.append([float(j) for j in w_list[1:]])

embeddings = np.array(embeddings)
os.chdir('/Users/sebastianma/Documents/CS224N/Project/')
pickle.dump(word2id, open('glove_word2id', 'wb'))
np.save('glove_embeddings', embeddings)


