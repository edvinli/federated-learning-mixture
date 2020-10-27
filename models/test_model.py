#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy

def test_img(net_g, datatest, args):
    with torch.no_grad():
        net_g.eval()
        #for i in range(len(net_l)):
        #    net_l[i].eval()
        # testing
        test_loss = 0
        test_loss_local = 0
        correct_g = 0
        correct_local = 0
        data_loader = DataLoader(datatest, batch_size=1)
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs_g = net_g(data)
            # sum up batch loss
            #test_loss += nn.NLLLoss(log_probs_g, target).item()
            # get the index of the max log-probability
            y_pred_g = log_probs_g.data.max(1, keepdim=True)[1]
            correct_g += y_pred_g.eq(target.data.view_as(y_pred_g)).long().cpu().sum() #Computes element-wise equality

        test_loss /= len(data_loader.dataset)
        #test_loss_local /= len(data_loader.dataset)
        accuracy_g = 100.00 * correct_g / len(data_loader.dataset)
        #accuracy_local = 100.00 * correct_local / len(data_loader.dataset)
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy_g.item(), test_loss


def test_img_mix(net_l, net_g, gate, datatest, args):
    with torch.no_grad():
        net_g.eval()
        net_l.eval()
        gate.eval()
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size = 1)
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        l = len(data_loader)
        g_w = []
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            gate_weight = gate(data)
            log_probs = gate_weight * net_l(data) + (1-gate_weight) * net_g(data)
            #test_loss += nn.NLLLoss(log_probs, target).item()
            y_pred = log_probs.data.max(1,keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

    return accuracy.item(), test_loss
            
