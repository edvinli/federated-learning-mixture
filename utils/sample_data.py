#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import itertools

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


def cifar_noniid2(dataset,dataset_test, num_users, p, n_data, n_data_test, overlap):
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
    
    idxs_test = np.arange(len(dataset_test),dtype=int)
    labels_test = np.array(dataset_test.targets)
    label_list_test = np.unique(dataset_test.targets)
    
    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1,:].argsort()]
    #print(idxs_labels)
    idxs_test = idxs_labels_test[0,:]
    idxs_test = idxs_test.astype(int)
    
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_classes = len(label_list)
    user_majority_labels = []
    overlap_list = list(itertools.combinations(range(num_classes), 2))
    for i in range(num_users):
    #Sample majority class for each user
        #if len(np.unique(labels[idxs])) > 1:
        if(overlap):
            majority_labels = list(itertools.product(range(10),repeat=2))[i]
        else:
            majority_labels = np.random.choice(np.unique(label_list), 2, replace = False)
        #else:
            #majority_labels[0] = np.unique(labels[idxs])
            #majority_labels[1] = np.unique(labels[idxs])

        label_list = np.array(list(set(label_list) - set(majority_labels)))
        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)
        
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p*n_data/2), replace = False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p*n_data/2), replace = False)

        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1))
        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs2))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))
        
        
        majority_labels1_idxs_test = idxs_test[majority_labels[0] == labels_test[idxs_test]]
        majority_labels2_idxs_test = idxs_test[majority_labels[1] == labels_test[idxs_test]]

        sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p*n_data_test/2), replace = False)
        sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p*n_data_test/2), replace = False)

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test))
        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs2_test))
        
        idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs1_test)))
        idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs2_test)))
        
    if p<1.0:
        for i in range(num_users):
            if(len(idxs)>=n_data):
                majority_labels = user_majority_labels[i]
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data), replace = False)
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs11))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
                
                non_majority_labels1_idxs_test = idxs_test[(majority_labels[0] != labels_test[idxs_test]) & (majority_labels[1] != labels_test[idxs_test])]
                sub_data_idxs11_test = np.random.choice(non_majority_labels1_idxs_test, int((1-p)*n_data_test), replace = False)
                dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs11_test))
                idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs11_test)))
                
            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))

    for i in range(num_users):
        print("Train")
        majority_labels = user_majority_labels[i]
        print("client %d %.2f " %(i, (sum(labels[dict_users[i]] == majority_labels[0])+sum(labels[dict_users[i]] == majority_labels[0]))/len(dict_users[i]) ))
        print(majority_labels)
        if i == range(num_users)[-1]:
            print(10*"-")

    for i in range(num_users):
        print("Test")
        majority_labels = user_majority_labels[i]
        print("client %d %.2f " %(i, (sum(labels_test[dict_users_test[i]] == majority_labels[0])+sum(labels_test[dict_users_test[i]] == majority_labels[0]))/len(dict_users_test[i]) ))
        print(majority_labels)
        if i == range(num_users)[-1]:
            print(10*"-")

    return dict_users, dict_users_test

#print(dataset[dict_users[0]])

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
