#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random
from torchvision import datasets, transforms
import itertools

def cifar_iid(dataset, num_users, n_data):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    #num_items = 500
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, int(n_data), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid2(dataset, num_users):
    """
    Sample IID client data, but different local data set size
    """
    tot_num_items = len(dataset)
    num_items_fraction = np.random.dirichlet(np.ones(num_users))
    num_items = np.round(tot_num_items * num_items_fraction).astype(int)
    while np.sum(num_items) > tot_num_items or (0 in num_items):
        num_items_fraction = np.random.dirichlet(np.ones(num_users))
        num_items = np.round(tot_num_items * num_items_fraction).astype(int)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def agnews_noniid(labels_train, labels_test, num_users, n_data, n_data_val, n_data_test, alpha):
    """
    Sample non-I.I.D client data from CIFAR dataset (dirichlet)
    :param dataset:
    :param num_users:
    :return:
    """
    idxs = np.arange(len(labels_train),dtype=int)
    labels = np.array(labels_train)
    label_list = np.unique(labels)
    n_classes = len(label_list)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_test = np.arange(len(labels_test),dtype=int)
    labels_test = np.array(labels_test)
    label_list_test = np.unique(labels_test)
    
    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1,:].argsort()]
    #print(idxs_labels)
    idxs_test = idxs_labels_test[0,:]
    idxs_test = idxs_test.astype(int)
    
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    num_classes = len(label_list)
    for c in range(num_users):
        print(c)
        idxs_test = np.arange(len(labels_test),dtype=int)
        label_distr = np.random.dirichlet(alpha*np.ones(n_classes)) #number of samples of each class
        label_distr = np.random.multinomial(n_data,label_distr)
    
        label_distr_val  = label_distr/sum(label_distr)*n_data_val
        label_distr_test = label_distr/sum(label_distr)*n_data_test
        label_distr_val = [int(x) for x in label_distr_val]
        label_distr_test = [int(x) for x in label_distr_test]
        for i in range(n_classes):
            if(label_distr[i]>0):
                #print(label_distr[i])
                sub_idx = np.random.choice(idxs[labels[idxs]==i], label_distr[i], replace=False) #sample class i
                dict_users[c] = np.concatenate( (dict_users[c], sub_idx) )
                idxs = np.array(list(set(idxs) - set(sub_idx)))
                
                sub_idx_val = np.random.choice(idxs[labels[idxs]==i], label_distr_val[i], replace=False)
                dict_users_val[c] = np.concatenate( (dict_users_val[c], sub_idx_val) )
                idxs = np.array(list(set(idxs) - set(sub_idx_val)))
                
                sub_idx_test = np.random.choice(idxs_test[labels_test[idxs_test]==i], label_distr_test[i], replace=False)
                dict_users_test[c] = np.concatenate( (dict_users_test[c], sub_idx_test) )

                #idxs_test = np.array(list(set(idxs_test) - set(sub_idx_test)))
            
    for c in range(num_users):
        print("Train")
        print(len(dict_users[c]))
        if c == range(num_users)[-1]:
            print(10*"-")
            
    for c in range(num_users):
        print("Test")
        print(len(dict_users_test[c]))
        if c == range(num_users)[-1]:
            print(10*"-")
            
    return dict_users, dict_users_val, dict_users_test



def agnews_noniid2(labels_train,labels_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
    idxs = np.arange(len(labels_train),dtype=int)
    labels = np.array(labels_train)
    label_list = np.unique(labels)
    n_classes = len(label_list)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_test = np.arange(len(labels_test),dtype=int)
    labels_test = np.array(labels_test)
    label_list_test = np.unique(labels_test)
    
    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1,:].argsort()]
    #print(idxs_labels)
    idxs_test = idxs_labels_test[0,:]
    idxs_test = idxs_test.astype(int)
    
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_classes = len(label_list)
    user_majority_labels = []
    overlap_list = list(itertools.combinations(range(num_classes), 2))

    for i in range(num_users):
    #Sample majority class for each user
        print(i)
        if(overlap):
            majority_labels = list(itertools.product(range(n_classes),repeat=2))
            majority_labels = random.choice(majority_labels)
        else:
            majority_labels = np.random.choice(np.unique(label_list), 2, replace = False)

            label_list = np.array(list(set(label_list) - set(majority_labels)))
        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)
        
        #train set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p*n_data/2), replace = False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p*n_data/2), replace = False)
        
        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1))
        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs2))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))
        
        #validation set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p*n_data_val/2), replace = False)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p*n_data_val/2), replace = False)
        
        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val))
        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs2_val))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))
        
        #test set
        majority_labels1_idxs_test = idxs_test[majority_labels[0] == labels_test[idxs_test]]
        majority_labels2_idxs_test = idxs_test[majority_labels[1] == labels_test[idxs_test]]

        sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p*n_data_test/2), replace = False)
        sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p*n_data_test/2), replace = False)

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test))
        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs2_test))
        
        #idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs1_test)))
        #idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs2_test)))
        
    if p<1.0:
        for i in range(num_users):
            if(len(idxs)>=n_data):
                majority_labels = user_majority_labels[i]
                #train set
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data), replace = False)
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs11))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
                
                #validation set
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11_val = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data_val), replace = False)
                dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs11_val))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
                
                #test set
                non_majority_labels1_idxs_test = idxs_test[(majority_labels[0] != labels_test[idxs_test]) & (majority_labels[1] != labels_test[idxs_test])]
                sub_data_idxs11_test = np.random.choice(non_majority_labels1_idxs_test, int((1-p)*n_data_test), replace = False)
                dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs11_test))
                #idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs11_test)))
                
            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))

    for i in range(num_users):
        print("Train")
        majority_labels = user_majority_labels[i]
        print("client %d %.2f %d " %(i, (sum(labels[dict_users[i]] == majority_labels[0])+sum(labels[dict_users[i]] == majority_labels[0]))/len(dict_users[i]),len(dict_users[i]) ))
        print(majority_labels)
        if i == range(num_users)[-1]:
            print(10*"-")

    for i in range(num_users):
        print("Test")
        majority_labels = user_majority_labels[i]
        print("client %d %.2f %d " %(i, (sum(labels_test[dict_users_test[i]] == majority_labels[0])+sum(labels_test[dict_users_test[i]] == majority_labels[0]))/len(dict_users_test[i]), len(dict_users_test[i]) ))
        print(majority_labels)
        if i == range(num_users)[-1]:
            print(10*"-")

    return dict_users, dict_users_val, dict_users_test

#print(dataset[dict_users[0]])

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
