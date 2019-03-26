"""
CS224N 2018-19: Project
proj.py : 2 d projection of [CLS] hidden state
Guoqin Ma <sebsk@stanford.edu>

Usage:
    proj.py PROJ [options]

Options:
    -h --help                               show this screen.
    --data=<file>                           dataset [default: df_train.csv]
    --cuda                                  use GPU
    --batch-size=<int>                      batch size [default: 32]
    --debug                                 use small datasets to debug
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from pytorch_pretrained_bert import BertTokenizer
from bert_model import DefaultModel
from bert_model import sents_to_tensor
from utils import batch_iter
import torch
import pandas as pd
import numpy as np
from docopt import docopt

args = docopt(__doc__)

device = torch.device("cuda:0" if args['--cuda'] else "cpu")

bert_tuned = DefaultModel.load('default_bert-base-uncased_model.bin', device=device)
bert = bert_tuned.bert.bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = int(args['--batch-size'])
bert_size = 'base'

bert.to(device)

df = pd.read_csv(args['--data'], index_col=0)

if args['--debug']:
    df = df.iloc[:batch_size]

bert.eval()

cls_hidden_states = []
labels = []

label_name = ['not related or not informative', 'other useful information', 'donations and volunteering',
              'affected individuals', 'sympathy and support', 'infrastructure and utilities damage',
              'caution and advice']

with torch.no_grad():
    for sents, targets in batch_iter(df, batch_size, shuffle=True, bert=bert_size):
        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(tokenizer, sents, device)
        encoded_layers, pooled_output = bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=False)
        cls_hidden_state = pooled_output.data.cpu().numpy()
        cls_hidden_states.append(cls_hidden_state)
        labels.extend(targets)

if args['--debug']:
    cls_hidden_states = cls_hidden_states[0]
else:
    cls_hidden_states = np.concatenate(cls_hidden_states)

labels = np.array(labels)

if args['PROJ'].upper() == 'PCA':
    pca = PCA(n_components=3)
    scaler = StandardScaler(with_mean=True, with_std=False)

    cls_scale = scaler.fit_transform(cls_hidden_states)
    cls_pc = pca.fit_transform(cls_scale)

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", 'chocolate']
    plt.figure(figsize=(20, 20))

    for i in range(7):
        plt.scatter(cls_pc[:, 0][labels==i], cls_pc[:, 1][labels==i], c=sns.xkcd_rgb[colors[i]], label=label_name[i],
                    alpha=0.3)
    plt.legend(fontsize='xx-large')
    plt.title('[CLS] hidden state', size=20)
    plt.savefig('cls_pca.png')

elif args['PROJ'].upper() == 'TSNE':
    tsne = TSNE()

    cls_tsne = tsne.fit_transform(cls_hidden_states)

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", 'chocolate']
    plt.figure(figsize=(20, 20))

    for i in range(7):
        plt.scatter(cls_tsne[:, 0][labels==i], cls_tsne[:, 1][labels==i], c=sns.xkcd_rgb[colors[i]], label=label_name[i],
                    alpha=0.3)
    plt.legend(fontsize='xx-large')
    plt.title('[CLS] hidden state', size=20)
    plt.savefig('cls_tsne.png')

elif args['PROJ'].upper() == 'ISOMAP':
    tsne = TSNE()

    cls_tsne = tsne.fit_transform(cls_hidden_states)

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", 'chocolate']
    plt.figure(figsize=(20, 20))

    for i in range(7):
        plt.scatter(cls_tsne[:, 0][labels==i], cls_tsne[:, 1][labels==i], c=sns.xkcd_rgb[colors[i]], label=label_name[i],
                    alpha=0.3)
    plt.legend(fontsize='xx-large')
    plt.title('[CLS] hidden state', size=20)
    plt.savefig('cls_tsne.png')








