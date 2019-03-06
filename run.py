#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
run.py : train, test, eval model.
Guoqin Ma <sebsk@stanford.edu>

Usage:
    run.py train [options]
    run.py test [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train=<file>                          train file [default: df_train.csv]
    --dev=<file>                            dev file [default: df_val.csv]
    --test=<file>                           test file [default: df_test.csv]
    --vocab=<file>                          vocab file from pickle [default: glove_word2id]
    --embeddings=<file>                     embedding matrix in .npy format [default: glove_embeddings.npy]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 64]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 100]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path [default: model.bin]
    --model-path=<file>                     model load path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 500]
    --dropout=<float>                       dropout [default: 0.3]
    --verbose                               whether to output the test results
"""

import pickle
import numpy as np
from vocab import VocabEntry
import torch
import pandas as pd
import time
import sys
from docopt import docopt
from baseline import BaselineModel
from utils import batch_iter
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def validation(model, df_val, loss_func, device):
    """ validation of model during training.
    @param model (nn.Module): the model being trained
    @param df_val (dataframe): validation dataset, sorted in descending text length
    @param loss_func(nn.Module): loss function

    @return avg loss value across validation dataset
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        sents = [text.split(' ') for text in df_val.ProcessedText]
        pre_softmax = model(sents)
        loss = loss_func(pre_softmax, torch.tensor(df_val.InformationType_label.values, dtype=torch.long, device=device))

    if was_training:
        model.train()

    return loss.item()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, path='cm', cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    pickle.dump(cm, open(path, 'wb'))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def train(args):

    label_name = ['not related or not informative', 'other useful information', 'donations and volunteering',
                  'affected individuals', 'sympathy and support', 'infrastructure and utilities damage',
                  'caution and advice']

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    start_time = time.time()
    print('Initializing Glove vocab and embeddings...', file=sys.stderr)
    glove_word2id = pickle.load(open(args['--vocab'], 'rb'))
    glove_word2id.update({'<unk>': len(glove_word2id)})
    glove_word2id.update({'<pad>': len(glove_word2id)})
    vocab = VocabEntry(glove_word2id)

    embedding_matrix = np.load(open(args['--embeddings'], 'rb'))
    embedding_matrix = np.vstack((embedding_matrix,
                                  np.random.uniform(embedding_matrix.min(), embedding_matrix.max(),
                                                    (2, embedding_matrix.shape[1]))))
    glove_embeddings = torch.tensor(embedding_matrix, dtype=torch.float, device=device)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    start_time = time.time()
    print('Importing data...', file=sys.stderr)
    df_train = pd.read_csv(args['--train'], index_col=0)
    df_val = pd.read_csv(args['--dev'], index_col=0)
    train_label = dict(df_train.InformationType_label.value_counts())
    label_max = float(max(train_label.values()))
    train_label_weight = torch.tensor([label_max/train_label[i] for i in range(len(train_label))], device=device)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    start_time = time.time()
    print('Set up model...', file=sys.stderr)

    model = BaselineModel(hidden_size=int(args['--hidden-size']), embedding=glove_embeddings,
                                   vocab=vocab, n_class=len(label_name), dropout_rate=float(args['--dropout']))
    model = model.to(device)
    print('Use device: %s' % device, file=sys.stderr)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight, reduction='mean')
    torch.save(cn_loss, 'loss_func')  # for later testing

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Begin Maximum Likelihood training...')

    while True:
        epoch += 1

        for sents, targets in batch_iter(df_train, batch_size=train_batch_size, shuffle=True):  # for each epoch
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(sents)

            pre_softmax = model(sents)

            loss = cn_loss(pre_softmax, torch.tensor(targets, dtype=torch.long, device=device))

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = loss.item() * batch_size
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, '
                      'cum. examples %d, speed %.2f examples/sec, '
                      'time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         cum_examples,
                                                                                         report_examples / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = 0.

                print('begin validation ...', file=sys.stderr)

                validation_loss = validation(model, df_val, cn_loss, device)   # dev batch size can be a bit larger

                print('validation: iter %d, loss %f' % (train_iter, validation_loss), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or validation_loss < min(hist_valid_scores)
                hist_valid_scores.append(validation_loss)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def test(args):

    label_name = ['not related or not informative', 'other useful information', 'donations and volunteering',
                  'affected individuals', 'sympathy and support', 'infrastructure and utilities damage',
                  'caution and advice']

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    print('load best model...')
    model = BaselineModel.load(args['--model-path'], device)
    model.to(device)

    model.eval()

    df_test = pd.read_csv(args['--test'], index_col=0)

    cn_loss = torch.load('loss_func')
    sents = [text.split(' ') for text in df_test.ProcessedText]

    with torch.no_grad():
        pre_softmax = model(sents)
        loss = cn_loss(pre_softmax, torch.tensor(df_test.InformationType_label.values, dtype=torch.long, device=device))

        softmax = torch.nn.Softmax(dim=1)
        prob = softmax(pre_softmax)
        prediction = [t.item() for t in list(torch.argmax(prob, dim=1))]

    pickle.dump([label_name[i] for i in prediction], open('test_prediction', 'wb'))

    accuracy = accuracy_score(df_test.InformationType_label.values, prediction)
    matthews = matthews_corrcoef(df_test.InformationType_label.values, prediction)

    precisions = {}
    recalls = {}
    f1s = {}
    aucrocs = {}

    for i in range(len(label_name)):
        prediction_ = [1 if pred == i else 0 for pred in prediction]
        true_ = [1 if label == i else 0 for label in df_test.InformationType_label.values]
        f1s.update({label_name[i]: f1_score(true_, prediction_)})
        precisions.update({label_name[i]: precision_score(true_, prediction_)})
        recalls.update({label_name[i]: recall_score(true_, prediction_)})
        aucrocs.update({label_name[i]: roc_auc_score(true_, list(t.item() for t in prob[:, i]))})

    metrics_dict = {'loss': loss, 'accuracy': accuracy, 'matthews coef': matthews, 'precision': precisions,
                         'recall': recalls, 'f1': f1s, 'aucroc': aucrocs}

    pickle.dump(metrics_dict, open('evaluation_metrics', 'wb'))

    cm = plot_confusion_matrix(list(df_test.InformationType_label.values), prediction, label_name, normalize=False,
                          path='test_confusion_matrix', title='confusion matrix for test dataset')
    plt.savefig('test_confusion_matrix', format='png')
    cm_norm = plot_confusion_matrix(list(df_test.InformationType_label.values), prediction, label_name, normalize=True,
                          path='test normalized_confusion_matrix', title='normalized confusion matrix for test dataset')
    plt.savefig('test_normalized_confusion_matrix', format='png')

    if args['--verbose']:
        print('loss: %.2f' % loss)
        print('accuracy: %.2f' % accuracy)
        print('matthews coef: %.2f' % matthews)
        for i in range(len(label_name)):
            print('precision score for %s: %.2f' % (label_name[i], precisions[label_name[i]]))
            print('recall score for %s: %.2f' % (label_name[i], recalls[label_name[i]]))
            print('f1 score for %s: %.2f' % (label_name[i], f1s[label_name[i]]))
            print('auc roc score for %s: %.2f' % (label_name[i], aucrocs[label_name[i]]))


if __name__ == '__main__':

    args = docopt(__doc__)

    if args['train']:
        train(args)

    elif args['test']:
        test(args)

    else:
        raise RuntimeError('invalid run mode')





