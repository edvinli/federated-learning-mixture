#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def emnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    num_items = 500
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users,p):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
     
    print(len(dict_users[0]))
    return dict_users

def mnist_noniid2(dataset, num_users, p):
    #n_data = int(len(dataset)/num_users) #data per client
    #n_data = 500
    n_data = 300
    
    idxs = np.arange(len(dataset),dtype=int)
    labels = dataset.train_labels.numpy()
    label_list = np.unique(labels)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    #Sample majority class for each user
    user_majority_labels = []
    for i in range(num_users):
        majority_labels = np.random.choice(label_list, 2, replace = False)
        user_majority_labels.append(majority_labels)

        #label_list = list(set(label_list) - set(majority_labels))

        majority_label_idxs = (majority_labels[0] == labels[idxs]) | (majority_labels[1] == labels[idxs])
        sub_data_idxs = np.random.choice(idxs[majority_label_idxs], int(p*n_data), replace = False)
        
        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

    if(p<1):
        for i in range(num_users):
            majority_labels = user_majority_labels[i]

            non_majority_label_idxs = (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])
            sub_data_idxs = np.random.choice(idxs[non_majority_label_idxs], int((1-p)*n_data), replace = False)

            dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
            idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

            print(sum(majority_labels[0] == labels[dict_users[i]])/len(labels[dict_users[i]]) + sum(majority_labels[1] == labels[dict_users[i]])/len(labels[dict_users[i]]))
            print(len(dict_users[i]))

    return dict_users

def mnist_iid2(dataset, num_users):
    """
    Sample IID client data, but w/ different local data set size
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

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    #num_items = 2000
    num_items = 500
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
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

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_noniid2(dataset, num_users, p, n_data):
    #n_data = int(len(dataset)/num_users) #data per client
    #n_data = 500
    #n_data_k = [100, 300, 500, 500, 600]
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    #Sample majority class for each user
    user_majority_labels = []
    for i in range(num_users):
        #n_data = n_data_k[i]
        majority_labels = np.random.choice(label_list, 2, replace = False)
        user_majority_labels.append(majority_labels)

        label_list = list(set(label_list) - set(majority_labels))

        #majority_label_idxs = (majority_labels[0] == labels[idxs]) | (majority_labels[1] == labels[idxs]) | (majority_labels[2] == labels[idxs])
        
        majority_label_idxs = (majority_labels[0] == labels[idxs]) | (majority_labels[1] == labels[idxs])        
        
        sub_data_idxs = np.random.choice(idxs[majority_label_idxs], int(p*n_data), replace = False)
        
        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

    if(p < 1.0):
        for i in range(num_users):
            majority_labels = user_majority_labels[i]

            #non_majority_label_idxs = (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs]) & (majority_labels[2] != labels[idxs])
            
            non_majority_label_idxs = (majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])
            
            sub_data_idxs = np.random.choice(idxs[non_majority_label_idxs], int((1-p)*n_data), replace = False)

            dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
            idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

            print(sum(majority_labels[0] == labels[dict_users[i]])/len(labels[dict_users[i]]) + sum(majority_labels[1] == labels[dict_users[i]])/len(labels[dict_users[i]]))
            print(len(dict_users[i]))

    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
