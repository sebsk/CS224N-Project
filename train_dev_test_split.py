#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
train_dev_test_split.py
Guoqin Ma <sebsk@stanford.edu>

Usage:
    train_dev_test_split.py [options]

Options:
    -h --help                               show this screen.
    --dev=<int>                             dev set size [default: 5000]
    --test=<int>                            test set size [default: 5000]
    --random-state=<int>                    random state for pandas dataframe shuffle [default: None]

"""

import pandas as pd
from docopt import docopt

args = docopt(__doc__)
df_tweets = pd.read_csv('en_disaster.csv', index_col=0)

"""
df_tweets contains preprocessed Tweets text. See "text_preprocessing.ipynb" for details
Stopwords remain.
No lemmatization.
Replicated post-processed Tweets are dropped.

columns: ['InformationSource', 'InformationType', 'InformationType_coarse',
       'InformationType_conf', 'InformationType_fine', 'Informativeness',
       'Informativeness_coarse', 'Informativeness_conf', 'ProcessedText',
       'Timestamp', 'TweetID', 'TweetText', 'TweetText_no_rt', 'event',
       'event_coarse', 'label', 'language', 'location', 'tweet_lat',
       'tweet_lon', 'year', 'EventInterested', 'Informativeness_label',
       'Informativeness_label_ft', 'binary_label', 'ProcessedText_length',
       'InformationType_label']
"""

# Commented section moved to Jupyter notebook
# df_tweets['ProcessedText_length'] = [len(text.split(' ')) for text in df_tweets.ProcessedText]
# df_tweets.sort_values(by='ProcessedText_length', ascending=False, inplace=True)

# label_dict = dict()
# for i, l in enumerate(list(df_tweets.InformationType_coarse.value_counts().keys())):
#     label_dict.update({l: i})
#
# df_tweets['InformationType_label'] = [label_dict[label] for label in df_tweets.InformationType_coarse]
#
# df_tweets.to_csv('en_disaster.csv')

try:
    random_state = int(args['--random-state'])
except:
    random_state = None

df_tweets = df_tweets.sample(frac=1, random_state=random_state)
n_val = int(args['--dev'])
n_test = int(args['--test'])
n_train = df_tweets.shape[0] - n_val - n_test
df_train = df_tweets.iloc[:n_train].sort_values(by='ProcessedText_length', ascending=False)
df_val = df_tweets.iloc[n_train:n_train + n_val].sort_values(by='ProcessedText_length', ascending=False)
df_test = df_tweets.iloc[n_train + n_val:].sort_values(by='ProcessedText_length', ascending=False)

df_train.to_csv('df_train.csv')
df_val.to_csv('df_val.csv')
df_test.to_csv('df_test.csv')