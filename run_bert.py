#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Project
run_bert.py : train, test, eval bert model.
Guoqin Ma <sebsk@stanford.edu>

Usage:
    run_bert.py train MODEL BERT_CONFIG [options]
    run_bert.py test MODEL BERT_CONFIG [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train=<file>                          train file [default: df_train.csv]
    --dev=<file>                            dev file [default: df_val.csv]
    --test=<file>                           test file [default: df_test.csv]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --hidden-size=<int>                     hidden size for lstm [default: 256]
    --out-channel=<int>                     out channel for cnn [default: 16]
    --clip-grad=<float>                     gradient clipping [default: 1.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 100]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 3]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr-bert=<float>                       BERT learning rate [default: 0.00002]
    --lr=<float>                            learning rate [default: 0.001]
    --valid-niter=<int>                     perform validation after how many iterations [default: 500]
    --dropout=<float>                       dropout [default: 0.3]
    --verbose                               whether to output the test results
"""

from pytorch_pretrained_bert import BertAdam
from bert_model import DefaultModel, CustomBertLSTMModel, CustomBertConvModel
import logging
import pickle
import numpy as np
import torch
import pandas as pd
import time
import sys
from docopt import docopt
from utils import batch_iter
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def validation(model, df_val, bert_size, loss_func, device):
    """ validation of model during training.
    @param model (nn.Module): the model being trained
    @param df_val (dataframe): validation dataset
    @param bert_size (str): large or base
    @param loss_func(nn.Module): loss function
    @param device (torch.device)

    @return avg loss value across validation dataset
    """
    was_training = model.training
    model.eval()

    df_val = df_val.sort_values(by='ProcessedText_BERT'+bert_size+'_length', ascending=False)

    ProcessedText_BERT = list(df_val.ProcessedText_BERT)
    InformationType_label = list(df_val.InformationType_label)

    val_batch_size = 32

    n_batch = int(np.ceil(df_val.shape[0]/val_batch_size))

    total_loss = 0.

    with torch.no_grad():
        for i in range(n_batch):
            sents = ProcessedText_BERT[i*val_batch_size: (i+1)*val_batch_size]
            targets = torch.tensor(InformationType_label[i*val_batch_size: (i+1)*val_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)
            pre_softmax = model(sents)
            batch_loss = loss_func(pre_softmax, targets)
            total_loss += batch_loss.item()*batch_size

    if was_training:
        model.train()

    return total_loss/df_val.shape[0]

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

    prefix = args['MODEL']+'_'+args['BERT_CONFIG']

    bert_size = args['BERT_CONFIG'].split('-')[1]

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

    if args['MODEL'] == 'default':
        model = DefaultModel(args['BERT_CONFIG'], device, len(label_name))
        optimizer = BertAdam([
                {'params': model.bert.bert.parameters()},
                {'params': model.bert.classifier.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    elif args['MODEL'] == 'lstm':
        model = CustomBertLSTMModel(args['BERT_CONFIG'], device, float(args['--dropout']), len(label_name),
                                    lstm_hidden_size=int(args['--hidden-size']))
        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.lstm.parameters(), 'lr': float(args['--lr'])},
                {'params': model.hidden_to_softmax.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    elif args['MODEL'] == 'cnn':
        model = CustomBertConvModel(args['BERT_CONFIG'], device, float(args['--dropout']), len(label_name),
                                    out_channel=int(args['--out-channel']))
        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.conv.parameters(), 'lr': float(args['--lr'])},
                {'params': model.hidden_to_softmax.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))

    model = model.to(device)
    print('Use device: %s' % device, file=sys.stderr)
    print('Done! time elapsed %.2f sec' % (time.time() - start_time), file=sys.stderr)
    print('-' * 80, file=sys.stderr)

    model.train()

    cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight, reduction='mean')
    torch.save(cn_loss, 'loss_func')  # for later testing

    train_batch_size = int(args['--batch-size'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = prefix+'_model.bin'

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Begin Maximum Likelihood training...')

    while True:
        epoch += 1

        for sents, targets in batch_iter(df_train, batch_size=train_batch_size, shuffle=True, bert=bert_size):  # for each epoch
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(sents)

            pre_softmax = model(sents)

            loss = cn_loss(pre_softmax, torch.tensor(targets, dtype=torch.long, device=device))

            loss.backward()

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

                validation_loss = validation(model, df_val, bert_size, cn_loss, device)   # dev batch size can be a bit larger

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
                        print('load previously best model and decay learning rate to %f%%' %
                              (float(args['--lr-decay'])*100), file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= float(args['--lr-decay'])

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def test(args):

    label_name = ['not related or not informative', 'other useful information', 'donations and volunteering',
                  'affected individuals', 'sympathy and support', 'infrastructure and utilities damage',
                  'caution and advice']

    prefix = args['MODEL'] + '_' + args['BERT_CONFIG']

    bert_size = args['BERT_CONFIG'].split('-')[1]

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    print('load best model...')

    if args['MODEL'] == 'default':
        model = DefaultModel.load(prefix + '_model.bin', device)
    elif args['MODEL'] == 'lstm':
        model = CustomBertLSTMModel.load(prefix+'_model.bin', device)
    elif args['MODEL'] == 'cnn':
        model = CustomBertConvModel.load(prefix+'_model.bin', device)

    model.to(device)

    model.eval()

    df_test = pd.read_csv(args['--test'], index_col=0)

    df_test = df_test.sort_values(by='ProcessedText_BERT'+bert_size+'_length', ascending=False)

    test_batch_size = 32

    n_batch = int(np.ceil(df_test.shape[0]/test_batch_size))

    cn_loss = torch.load('loss_func', map_location=lambda storage, loc: storage).to(device)

    ProcessedText_BERT = list(df_test.ProcessedText_BERT)
    InformationType_label = list(df_test.InformationType_label)

    test_loss = 0.
    prediction = []
    prob = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i in range(n_batch):
            sents = ProcessedText_BERT[i*test_batch_size: (i+1)*test_batch_size]
            targets = torch.tensor(InformationType_label[i * test_batch_size: (i + 1) * test_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)

            pre_softmax = model(sents)
            batch_loss = cn_loss(pre_softmax, targets)
            test_loss += batch_loss.item()*batch_size
            prob_batch = softmax(pre_softmax)
            prob.append(prob_batch)

            prediction.extend([t.item() for t in list(torch.argmax(prob_batch, dim=1))])

    prob = torch.cat(tuple(prob), dim=0)
    loss = test_loss/df_test.shape[0]

    pickle.dump([label_name[i] for i in prediction], open(prefix+'_test_prediction', 'wb'))

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

    pickle.dump(metrics_dict, open(prefix+'_evaluation_metrics', 'wb'))

    cm = plot_confusion_matrix(list(df_test.InformationType_label.values), prediction, label_name, normalize=False,
                          path=prefix+'_test_confusion_matrix', title='confusion matrix for test dataset')
    plt.savefig(prefix+'_test_confusion_matrix', format='png')
    cm_norm = plot_confusion_matrix(list(df_test.InformationType_label.values), prediction, label_name, normalize=True,
                          path=prefix+'_test normalized_confusion_matrix', title='normalized confusion matrix for test dataset')
    plt.savefig(prefix+'_test_normalized_confusion_matrix', format='png')

    if args['--verbose']:
        print('loss: %.2f' % loss)
        print('accuracy: %.2f' % accuracy)
        print('matthews coef: %.2f' % matthews)
        print('-' * 80)
        for i in range(len(label_name)):
            print('precision score for %s: %.2f' % (label_name[i], precisions[label_name[i]]))
            print('recall score for %s: %.2f' % (label_name[i], recalls[label_name[i]]))
            print('f1 score for %s: %.2f' % (label_name[i], f1s[label_name[i]]))
            print('auc roc score for %s: %.2f' % (label_name[i], aucrocs[label_name[i]]))
            print('-' * 80)


if __name__ == '__main__':

    args = docopt(__doc__)

    logging.basicConfig(level=logging.INFO)

    if args['train']:
        train(args)

    elif args['test']:
        test(args)

    else:
        raise RuntimeError('invalid run mode')
