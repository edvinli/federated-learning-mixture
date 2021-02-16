import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch import nn
import torch.nn.functional as F
import torchtext

def evaluate_validation(scores, loss_function, gold):
    guesses = scores.argmax(dim=1)
    n_correct = (guesses == gold).sum().item()
    return n_correct, loss_function(scores, gold).item()


class LanguageClientUpdate(object):
    def __init__(self, args, train_set=None,  test_set=None, idxs_train=None, 
                 idxs_val=None,idxs_test=None, TEXT=None, LABEL=None):
        self.args = args
        self.loss_function = nn.NLLLoss()
        
        #TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
        #LABEL = torchtext.data.LabelField(is_target=True)
        datafields = [('text', TEXT), ('label', LABEL)]    

        train_examples = [train_set.examples[i] for i in idxs_train]
        val_examples = [train_set.examples[i] for i in idxs_val]
        test_examples = [test_set.examples[i] for i in idxs_test]
        
        local_train_set = torchtext.data.Dataset(train_examples, datafields)
        local_val_set = torchtext.data.Dataset(val_examples, datafields)
        local_test_set = torchtext.data.Dataset(test_examples, datafields)
        
        self.n_val = len(local_val_set)
        self.n_test = len(local_test_set)
        
        #TEXT.build_vocab(local_train_set, max_size=10000)
        #LABEL.build_vocab(local_train_set)

        train_iterator = torchtext.data.BucketIterator(
            local_train_set,
            device=args.device,
            batch_size=args.local_bs,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=True) 

        val_iterator = torchtext.data.BucketIterator(
            local_val_set,
            device=args.device,
            batch_size=args.local_bs,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=False,
            sort=True)
        
        test_iterator = torchtext.data.Iterator(
            local_test_set,
            device=args.device,
            batch_size=args.local_bs,
            repeat=False,
            train=False,
            sort=False)
        
        self.train_batches = list(train_iterator)
        self.val_batches = list(val_iterator)
        self.test_batches = list(test_iterator)
        
    def train(self, net, n_epochs, learning_rate):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        epoch_loss = []
        for iter in range(n_epochs):
            net.train()
            loss_sum = 0
            n_batches = 0
            for batch in self.train_batches:

                scores = net(batch.text)
                loss = self.loss_function(scores, batch.label)

                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                n_batches += 1

            train_loss = loss_sum / n_batches
            epoch_loss.append(train_loss)
            #history['train_loss'].append(train_loss)
            
        return net.state_dict(), epoch_loss[-1]
    
    def train_finetune(self, net, n_epochs, learning_rate, val):
        # train and update
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        epoch_loss = []
        patience = 10
        model_best = net.state_dict()
        train_acc_best = np.inf
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        for iter in range(n_epochs):
            net.train()
            loss_sum = 0
            n_batches = 0
            for batch in self.train_batches:

                scores = net(batch.text)
                loss = self.loss_function(scores, batch.label)

                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                n_batches += 1

            train_loss = loss_sum / n_batches
            epoch_loss.append(train_loss)
            if(iter%5==0):
                val_acc, val_loss = self.validate(net,val)
                net.train()
                if(val_loss < val_loss_best - 0.01):
                    counter = 0
                    model_best = net.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    print("Iter %d | %.2f" %(iter, val_acc_best))
                else:
                    counter = counter + 1

                if counter == patience:
                    return model_best, val_loss_best, val_acc_best, train_acc_best
            
        return model_best, val_loss_best, val_acc_best
    
    
    def train_mix(self, net_local, net_global, gate, train_gate_only, n_epochs, early_stop, learning_rate, val):
        # train and update
        if(train_gate_only):
            optimizer = torch.optim.Adam(list(gate.parameters()),lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(list(net_local.parameters()) +  list(gate.parameters()),lr=learning_rate)
        epoch_loss = []
        patience = 10
        gate_best = gate.state_dict()
        local_best = net_local.state_dict()
        global_best = net_global.state_dict()
        train_acc_best = np.inf
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        for iter in range(n_epochs):
            net_local.train()
            net_global.train()
            gate.train()
            loss_sum = 0
            n_batches = 0
            for batch in self.train_batches:

                scores_l = net_local(batch.text)
                scores_g = net_global(batch.text)
                gate_weight = gate(batch.text)
                scores = gate_weight * scores_l + (1-gate_weight) * scores_g
                loss = self.loss_function(scores, batch.label)

                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                n_batches += 1

            train_loss = loss_sum / n_batches
            epoch_loss.append(train_loss)
            if(iter%5==0):
                val_acc, val_loss = self.validate_mix(net_local, net_global, gate, val)
                net_local.train()
                net_global.train()
                gate.train()
                if(val_loss < val_loss_best - 0.01):
                    counter = 0
                    gate_best = gate.state_dict()
                    local_best = net_local.state_dict()
                    global_best = net_global.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    print("Iter %d | %.2f" %(iter, val_acc_best))
                else:
                    counter = counter + 1

                if counter == patience:
                    return gate_best, local_best, global_best, val_loss_best, val_acc_best
            
        return gate_best, local_best, global_best, val_loss_best, val_acc_best
 
    
    def validate(self,net,val):
        net.eval()
        if(val):
            batches = self.val_batches
            dataset_size = self.n_val
        else:
            batches = self.test_batches
            dataset_size = self.n_test
        with torch.no_grad():
            loss_sum = 0
            n_correct = 0
            n_batches = 0
            for batch in batches:
                scores = net(batch.text)
                n_corr_batch, loss_batch = evaluate_validation(scores, self.loss_function, batch.label)
                loss_sum += loss_batch
                n_correct += n_corr_batch
                n_batches += 1
            val_acc = 100*n_correct / dataset_size
            val_loss = loss_sum / n_batches
        
        return val_acc, val_loss

    
    def validate_mix(self,net_l, net_g, gate, val):
        net_l.eval()
        net_g.eval()
        gate.eval()
        if(val):
            batches = self.val_batches
            dataset_size = self.n_val
        else:
            batches = self.test_batches
            dataset_size = self.n_test
        with torch.no_grad():
            loss_sum = 0
            n_correct = 0
            n_batches = 0
            for batch in batches:
                scores_l = net_l(batch.text)
                scores_g = net_g(batch.text)
                gate_weight = gate(batch.text)
                scores = gate_weight * scores_l + (1-gate_weight) * scores_g
                n_corr_batch, loss_batch = evaluate_validation(scores, self.loss_function, batch.label)
                loss_sum += loss_batch
                n_correct += n_corr_batch
                n_batches += 1
            val_acc = 100*n_correct / dataset_size
            val_loss = loss_sum / n_batches
        
        return val_acc, val_loss