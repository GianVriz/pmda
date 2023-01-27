#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle
import numpy as np
import os
import math
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

import src.data as data

from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from src.detm import DETM
from src.file_io import load_embeddings
from src.utils import nearest_neighbors, get_topic_coherence

def main_DETM(dataset, data_path, emb_file, save_path, model_file, batch_size=1000,
              num_topics=50, rho_size=300, emb_size=300, t_hidden_size=800, theta_act='relu', train_embeddings=1, eta_nlayers=3, eta_hidden_size=200, delta=0.005,
              lr=0.005, lr_factor=4.0, epochs=100, mode='train', optimizer='adam', seed=28, enc_drop=0.0, eta_dropout=0.0, clip=0.0, nonmono=10, wdecay=1.2e-6, anneal_lr=0, bow_norm=1,
              num_words=20, log_interval=10, visualize_every=1, eval_batch_size=1000, load_from='', tc=0):
    """
    Args:
    ----------------   data and file related arguments
    dataset          : name of corpus (str)
    data_path        : directory containing data (str)
    emb_file         : word embeddings file, used if train_embeddings=False (str)
    save_path        : path to save results (str)
    model_file       : model file, saved in save_path (str)
    batch_size       : number of documents in a batch for training (int)
    ----------------   model-related arguments
    num_topics       : number of topics (int)
    rho_size         : dimension of rho (int)
    emb_size         : dimension of embeddings (int)
    t_hidden_size    : dimension of hidden space of q(theta) (int)
    theta_act        : tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu (str)
    train_embeddings : whether to fix rho or train it (int)
    eta_nlayers      : number of layers for eta (int)
    eta_hidden_size  : number of hidden units for rnn (int)
    delta            : prior variance (float)
    ----------------   optimization-related arguments
    lr               : learning rate (float)
    lr_factor        : divide learning rate by this (float)
    epochs           : number of epochs to train (int)
    mode             : train or eval model (str)
    optimizer        : choice of optimizer (str)
    seed             : random seed (default: 28) (int)
    enc_drop         : dropout rate on encoder (float)
    eta_dropout      : dropout rate on rnn for eta (float)
    clip             : gradient clipping (float)
    nonmono          : number of bad hits allowed (int)
    wdecay           : some l2 regularization (float)
    anneal_lr        : whether to anneal the learning rate or not (int)
    bow_norm         : normalize the bows or not (int)
    ----------------   evaluation, visualization, and logging-related arguments
    num_words        : number of words for topic viz (int)
    log_interval     : when to log training (int)
    visualize_every  : when to visualize results (int)
    eval_batch_size  : input batch size for evaluation (int)
    load_from        : the name of the ckpt to eval from (str)
    tc               : whether to compute tc or not (int)
    """

    pca = PCA(n_components=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## set seed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    print('\n')
    print('=*'*100)
    print('Training a Dynamic Embedded Topic Model on ' + dataset.upper())
    print('=*'*100)

    ## get data
    # 1. vocabulary
    print('Getting vocabulary ...')
    data_file = data_path
    vocab, train, valid, test = data.get_data(data_file, temporal=True)
    vocab_size = len(vocab)

    # 1. training data
    print('Getting training data ...')
    train_tokens = train['tokens']
    train_counts = train['counts']
    train_times = train['times']
    num_times = len(np.unique(train_times))
    num_docs_train = len(train_tokens)
    train_rnn_inp = data.get_rnn_input(train_tokens, train_counts, train_times, num_times, vocab_size, num_docs_train)

    # 2. dev set
    print('Getting validation data ...')
    valid_tokens = valid['tokens']
    valid_counts = valid['counts']
    valid_times = valid['times']
    num_docs_valid = len(valid_tokens)
    valid_rnn_inp = data.get_rnn_input(valid_tokens, valid_counts, valid_times, num_times, vocab_size, num_docs_valid)

    # 3. test data
    print('Getting testing data ...')
    test_tokens = test['tokens']
    test_counts = test['counts']
    test_times = test['times']
    num_docs_test = len(test_tokens)
    test_rnn_inp = data.get_rnn_input(test_tokens, test_counts, test_times, num_times, vocab_size, num_docs_test)

    test_1_tokens = test['tokens_1']
    test_1_counts = test['counts_1']
    test_1_times = test_times
    num_docs_test_1 = len(test_1_tokens)
    test_1_rnn_inp = data.get_rnn_input(test_1_tokens, test_1_counts, test_1_times, num_times, vocab_size, num_docs_test)

    test_2_tokens = test['tokens_2']
    test_2_counts = test['counts_2']
    test_2_times = test_times
    num_docs_test_2 = len(test_2_tokens)
    test_2_rnn_inp = data.get_rnn_input(test_2_tokens, test_2_counts, test_2_times, num_times, vocab_size, num_docs_test)

    ## get embeddings
    print('Getting embeddings ...')
    embeddings = None
    if not train_embeddings:
        embeddings = load_embeddings(emb_file, emb_size, vocab)
        embeddings = torch.from_numpy(embeddings).to(device)
        embeddings_dim = embeddings.size()

    ## define checkpoint
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if mode == 'eval':
        ckpt = load_from
    else:
        ckpt = Path.cwd().joinpath(save_path, model_file)

    ## define model and optimizer
    if load_from != '':
        print('Loading checkpoint from {}'.format(load_from))
        with open(load_from, 'rb') as f:
            model = torch.load(f)
    else:
        model = DETM(num_topics, num_times, vocab_size, t_hidden_size, eta_hidden_size,
                     rho_size, emb_size, enc_drop, eta_nlayers, delta, train_embeddings,
                     theta_act, eta_dropout, embeddings)
    print('\nDETM architecture: {}'.format(model))
    model.to(device)

    optimizer = model.get_optimizer(optimizer, lr, wdecay)

    def train(epoch):
        """Train DETM on data for one epoch.
        """
        model.train()
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        indices = torch.randperm(num_docs_train)
        indices = torch.split(indices, batch_size)
        for idx, ind in enumerate(indices):
            optimizer.zero_grad()
            model.zero_grad()
            data_batch, times_batch = data.get_batch(train_tokens, train_counts, ind, vocab_size, emb_size, temporal=True, times=train_times)
            sums = data_batch.sum(1).unsqueeze(1)
            if bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch

            loss, nll, kl_alpha, kl_eta, kl_theta = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, num_docs_train)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            acc_loss += torch.sum(loss).item()
            acc_nll += torch.sum(nll).item()
            acc_kl_theta_loss += torch.sum(kl_theta).item()
            acc_kl_eta_loss += torch.sum(kl_eta).item()
            acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += 1

            if idx % log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_nll = round(acc_nll / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_kl_eta = round(acc_kl_eta_loss / cnt, 2)
                cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
                lr = optimizer.param_groups[0]['lr']
                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))

        cur_loss = round(acc_loss / cnt, 2)
        cur_nll = round(acc_nll / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2)
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
        lr = optimizer.param_groups[0]['lr']
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
        print('*'*100)

    def _eta_helper(rnn_inp):
        inp = model.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = model.init_hidden()
        output, _ = model.q_eta(inp, hidden)
        output = output.squeeze()
        etas = torch.zeros(model.num_times, model.num_topics).to(device)
        inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
        etas[0] = model.mu_q_eta(inp_0)
        for t in range(1, model.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            etas[t] = model.mu_q_eta(inp_t)
        return etas

    def get_eta(source):
        model.eval()
        with torch.no_grad():
            if source == 'val':
                rnn_inp = valid_rnn_inp
                return _eta_helper(rnn_inp)
            else:
                rnn_1_inp = test_1_rnn_inp
                return _eta_helper(rnn_1_inp)

    def get_theta(eta, bows):
        model.eval()
        with torch.no_grad():
            inp = torch.cat([bows, eta], dim=1)
            q_theta = model.q_theta(inp)
            mu_theta = model.mu_q_theta(q_theta)
            theta = F.softmax(mu_theta, dim=-1)
            return theta

    def get_completion_ppl(source):
        """Returns document completion perplexity.
        """
        model.eval()
        with torch.no_grad():
            alpha = model.mu_q_alpha
            if source == 'val':
                indices = torch.split(torch.tensor(range(num_docs_valid)), eval_batch_size)
                tokens = valid_tokens
                counts = valid_counts
                times = valid_times

                eta = get_eta('val')

                acc_loss = 0
                cnt = 0
                for idx, ind in enumerate(indices):
                    data_batch, times_batch = data.get_batch(tokens, counts, ind, vocab_size, emb_size, temporal=True, times=times)
                    sums = data_batch.sum(1).unsqueeze(1)
                    if bow_norm:
                        normalized_data_batch = data_batch / sums
                    else:
                        normalized_data_batch = data_batch

                    eta_td = eta[times_batch.type('torch.LongTensor')]
                    theta = get_theta(eta_td, normalized_data_batch)
                    alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
                    beta = model.get_beta(alpha_td).permute(1, 0, 2)
                    loglik = theta.unsqueeze(2) * beta
                    loglik = loglik.sum(1)
                    loglik = torch.log(loglik)
                    nll = -loglik * data_batch
                    nll = nll.sum(-1)
                    loss = nll / sums.squeeze()
                    loss = loss.mean().item()
                    acc_loss += loss
                    cnt += 1
                cur_loss = acc_loss / cnt
                ppl_all = round(math.exp(cur_loss), 1)
                print('*'*100)
                print('{} PPL: {}'.format(source.upper(), ppl_all))
                print('*'*100)
                return ppl_all
            else:
                indices = torch.split(torch.tensor(range(num_docs_test)), eval_batch_size)
                tokens_1 = test_1_tokens
                counts_1 = test_1_counts

                tokens_2 = test_2_tokens
                counts_2 = test_2_counts

                eta_1 = get_eta('test')

                acc_loss = 0
                cnt = 0
                indices = torch.split(torch.tensor(range(num_docs_test)), eval_batch_size)
                for idx, ind in enumerate(indices):
                    data_batch_1, times_batch_1 = data.get_batch(tokens_1, counts_1, ind, vocab_size, emb_size, temporal=True, times=test_times)
                    sums_1 = data_batch_1.sum(1).unsqueeze(1)
                    if bow_norm:
                        normalized_data_batch_1 = data_batch_1 / sums_1
                    else:
                        normalized_data_batch_1 = data_batch_1

                    eta_td_1 = eta_1[times_batch_1.type('torch.LongTensor')]
                    theta = get_theta(eta_td_1, normalized_data_batch_1)

                    data_batch_2, times_batch_2 = data.get_batch(tokens_2, counts_2, ind, vocab_size, emb_size, temporal=True, times=test_times)
                    sums_2 = data_batch_2.sum(1).unsqueeze(1)

                    alpha_td = alpha[:, times_batch_2.type('torch.LongTensor'), :]
                    beta = model.get_beta(alpha_td).permute(1, 0, 2)
                    loglik = theta.unsqueeze(2) * beta
                    loglik = loglik.sum(1)
                    loglik = torch.log(loglik)
                    nll = -loglik * data_batch_2
                    nll = nll.sum(-1)
                    loss = nll / sums_2.squeeze()
                    loss = loss.mean().item()
                    acc_loss += loss
                    cnt += 1
                cur_loss = acc_loss / cnt
                ppl_dc = round(math.exp(cur_loss), 1)
                print('*'*100)
                print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
                print('*'*100)
                return ppl_dc

    def _diversity_helper(beta, num_tops):
        list_w = np.zeros((num_topics, num_tops))
        for k in range(num_topics):
            gamma = beta[k, :]
            top_words = gamma.cpu().numpy().argsort()[-num_tops:][::-1]
            list_w[k, :] = top_words
        list_w = np.reshape(list_w, (-1))
        list_w = list(list_w)
        n_unique = len(np.unique(list_w))
        diversity = n_unique / (num_topics * num_tops)
        return diversity

    def get_topic_quality():
        """Returns topic coherence and topic diversity.
        """
        model.eval()
        with torch.no_grad():
            alpha = model.mu_q_alpha
            beta = model.get_beta(alpha)
            print('beta: ', beta.size())

            print('\n')
            print('#'*100)
            print('Get topic diversity...')
            num_tops = 25
            TD_all = np.zeros((num_times,))
            for tt in range(num_times):
                TD_all[tt] = _diversity_helper(beta[:, tt, :], num_tops)
            TD = np.mean(TD_all)
            print('Topic Diversity is: {}'.format(TD))

            print('\n')
            print('Get topic coherence...')
            print('train_tokens: ', train_tokens[0])
            TC_all = []
            cnt_all = []
            for tt in range(num_times):
                tc, cnt = get_topic_coherence(beta[:, tt, :].cpu().numpy(), train_tokens, vocab, temporal=True)
                TC_all.append(tc)
                cnt_all.append(cnt)
            print('TC_all: ', TC_all)
            TC_all = torch.tensor(TC_all)
            print('TC_all: ', TC_all.size())
            print('\n')
            print('Get topic quality...')
            quality = tc * diversity
            print('Topic Quality is: {}'.format(quality))
            print('#'*100)

    if mode == 'train':
        ## train model on data by looping through multiple epochs
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        for epoch in range(1, epochs):
            train(epoch)
            if epoch % visualize_every == 0:
                self.visualize(num_words, vocabulary)
            val_ppl = get_completion_ppl('val')
            print('val_ppl: ', val_ppl)
            if val_ppl < best_val_ppl:
                with open(ckpt, 'wb') as f:
                    torch.save(model, f)
                best_epoch = epoch
                best_val_ppl = val_ppl
            else:
                ## check whether to anneal lr
                lr = optimizer.param_groups[0]['lr']
                if anneal_lr and (len(all_val_ppls) > nonmono and val_ppl > min(all_val_ppls[:-nonmono]) and lr > 1e-5):
                    optimizer.param_groups[0]['lr'] /= lr_factor
            all_val_ppls.append(val_ppl)
        with open(ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            print('saving topic matrix beta...')
            alpha = model.mu_q_alpha
            beta = model.get_beta(alpha).cpu().numpy()
            scipy.io.savemat(ckpt+'_beta.mat', {'values': beta}, do_compression=True)
            if train_embeddings:
                print('saving word embedding matrix rho...')
                rho = model.rho.weight.cpu().numpy()
                scipy.io.savemat(ckpt+'_rho.mat', {'values': rho}, do_compression=True)
            print('computing validation perplexity...')
            val_ppl = get_completion_ppl('val')
            print('computing test perplexity...')
            test_ppl = get_completion_ppl('test')
    else:
        with open(ckpt, 'rb') as f:
            model = torch.load(f)
        model = model.to(device)

        print('saving alpha...')
        with torch.no_grad():
            alpha = model.mu_q_alpha.cpu().numpy()
            scipy.io.savemat(ckpt+'_alpha.mat', {'values': alpha}, do_compression=True)

        print('computing validation perplexity...')
        val_ppl = get_completion_ppl('val')
        print('computing test perplexity...')
        test_ppl = get_completion_ppl('test')
        print('computing topic coherence and topic diversity...')
        get_topic_quality()
        print('visualizing topics and embeddings...')
        self.visualize(num_words, vocabulary)
