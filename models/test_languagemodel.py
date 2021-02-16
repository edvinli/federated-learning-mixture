#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchtext
import numpy as np
import copy

def evaluate_validation(scores, loss_function, gold):
    guesses = scores.argmax(dim=1)
    n_correct = (guesses == gold).sum().item()
    return n_correct, loss_function(scores, gold).item()

def test_img(net, datatest, datafields, args):
    idxs_test = np.random.choice(range(len(datatest)),1000,replace=False)
    test_examples = [datatest.examples[i] for i in idxs_test]
    test_set = torchtext.data.Dataset(test_examples, datafields)
    loss_function = nn.NLLLoss()
    
    test_iterator = torchtext.data.BucketIterator(
        test_set,
        device=args.device,
        batch_size=args.local_bs,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=False,
        sort=True)
    
    test_batches = list(test_iterator)
    n_val = len(datatest)
    net.eval()
    with torch.no_grad():
        loss_sum = 0
        n_correct = 0
        n_batches = 0
        for batch in test_batches:
            scores = net(batch.text)
            n_corr_batch, loss_batch = evaluate_validation(scores, loss_function, batch.label)
            loss_sum += loss_batch
            n_correct += n_corr_batch
            n_batches += 1
        val_acc = 100*n_correct / n_val
        val_loss = loss_sum / n_batches

    return val_acc, val_loss

def test_img_mix(net_l, net_g, gate, datatest, datafields, args):
    idxs_test = np.random.choice(range(len(datatest)),1000,replace=False)
    test_examples = [datatest.examples[i] for i in idxs_test]
    test_set = torchtext.data.Dataset(test_examples, datafields)
    
    net_l.eval()
    net_g.eval()
    gate.eval()
    loss_function = nn.NLLLoss()
    
    test_iterator = torchtext.data.BucketIterator(
        test_set,
        device=args.device,
        batch_size=args.local_bs,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=False,
        sort=True)
    
    test_batches = list(test_iterator)
    n_val = len(datatest)
    with torch.no_grad():
        loss_sum = 0
        n_correct = 0
        n_batches = 0
        for batch in test_batches:
            scores_l = net_l(batch.text)
            scores_g = net_g(batch.text)
            gate_weight = gate(batch.text)
            scores = gate_weight * scores_l + (1-gate_weight)*scores_g
            n_corr_batch, loss_batch = evaluate_validation(scores, loss_function, batch.label)
            loss_sum += loss_batch
            n_correct += n_corr_batch
            n_batches += 1
        val_acc = 100*n_correct / n_val
        val_loss = loss_sum / n_batches

    return val_acc, val_loss