import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
import pandas as pd
from utils.sample_data import mnist_iid, mnist_iid2, mnist_noniid2, cifar_iid, cifar_iid2, cifar_noniid, cifar_noniid2
from utils.arguments import args_parser
from models.ClientUpdate import ClientUpdate
from models.Models import MLP, CNNCifar, GateCNN, GateMLP, CNNFashion, GateCNNFashion,GateCNNSoftmax, MLP2
from models.FederatedAveraging import FedAvg
from models.test_model import test_img, test_img_mix
import os.path
from sys import exit


if __name__ == '__main__':
    filename = 'results'
    filexist = os.path.isfile('save/'+filename) 
    if(not filexist):
        with open('save/'+filename,'a') as f1:
            f1.write('dataset;model;epochs;local_ep;num_clients;iid;p;opt;n_data;train_frac;train_gate_only;val_acc_avg_e2e;val_acc_avg_locals;val_acc_avg_fedavg;ft_val_acc;val_acc_avg_3;val_acc_avg_rep;val_acc_avg_repft;acc_test_mix;acc_test_locals;acc_test_fedavg;ft_test_acc;ft_train_acc;train_acc_avg_locals;run')

            f1.write('\n')
        
    args=args_parser()
    for run in range(args.runs):

        args.device = torch.device('cuda:{}'.format(args.gpu))
        
        #Create datasets
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist) 
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_clients)
            else:
                dict_users = mnist_noniid2(dataset_train, args.num_clients, args.p)
            
        elif args.dataset == 'cifar10':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
            
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_clients)
            else:
                dict_users = cifar_noniid2(dataset_train, args.num_clients, args.p, args.n_data)
                
        elif args.dataset == 'cifar100':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
            
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_clients)
            else:
                dict_users = cifar_noniid2(dataset_train, args.num_clients, args.p,args.n_data)
                
        elif args.dataset == 'fashion-mnist':
            trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download = True, transform = trans_fashionmnist)
            dataset_test = datasets.FashionMNIST('../data/fashion-mnist',train=False, download = True, transform = trans_fashionmnist)
            
            if args.iid:
                dict_users = cifar_iid(dataset_train,args.num_clients)
            else:
                dict_users = cifar_noniid2(dataset_train, args.num_clients, args.p, args.n_data)
        else:
            exit('error: dataset not available')
            
        img_size = dataset_train[0][0].shape

        input_length = 1
        for x in img_size:
            input_length *= x
            
        if (args.model == 'cnn') and (args.dataset in ['cifar10','cifar100']):
            net_glob_fedAvg = CNNCifar(args=args).to(args.device)
            gate_rep = MLP2(dim_in = 84*2 , dim_hidden = 84, dim_out=args.num_classes).to(args.device)
            gate_repft = MLP2(dim_in = 84*2 , dim_hidden = 84, dim_out=args.num_classes).to(args.device)
            gates_3 = GateCNNSoftmax(args=args).to(args.device)
            gates_e2e = GateCNN(args=args).to(args.device)
            net_locals = CNNCifar(args=args).to(args.device)

            #opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(range(args.num_clients), size = int(args.opt*args.num_clients), replace=False)
            opt[opt_out] = 0.0
                
        elif (args.model == 'cnn') and (args.dataset in ['mnist', 'fashion-mnist']):
            net_glob_fedAvg = CNNFashion(args=args).to(args.device)

            #gates = []
            gates_3 = GateCNNSoftmax(args=args).to(args.device)
            gates_e2e = GateCNN(args=args).to(args.device)
            net_locals = CNNCifar(args=args).to(args.device)

            #opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(range(args.num_clients), size = int(args.opt*args.num_clients), replace=False)
            opt[opt_out] = 0.0

        elif args.model == 'mlp':
            net_glob_fedAvg = MLP(dim_in=input_length, dim_hidden=200, dim_out=args.num_classes).to(args.device)

            #gates = []
            net_locals = MLP(dim_in=input_length, dim_hidden=200, dim_out=args.num_classes).to(args.device)
            gates_e2e = GateMLP(dim_in = input_length,dim_hidden=200, dim_out=1).to(args.device)
            gates_3 = GateMLPSoftmax(dim_in = input_length,dim_hidden=200, dim_out=3).to(args.device)
            #opt-out fraction
            opt = np.ones(args.num_clients)
            opt_out = np.random.choice(range(args.num_clients), size = int(args.opt*args.num_clients), replace=False)
            opt[opt_out] = 0.0
                
        else:
            exit('error: no such model')
        
        print(net_glob_fedAvg)

        gates_e2e.train()
        gates_3.train()
        net_locals.train()
        net_glob_fedAvg.train()

        # training
        acc_test_locals, acc_test_mix, acc_test_fedavg = [], [], []
        
        acc_test_finetuned_avg = []

        for iter in range(args.epochs):
            print('Round {:3d}'.format(iter))
            
            w_fedAvg = []
            alpha = []

            for idx in range(args.num_clients):
                print("FedAvg client %d" %(idx))
                
                client = ClientUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                
                if(opt[idx]):
                    #train FedAvg
                    w_glob_fedAvg, _ = client.train(net = copy.deepcopy(net_glob_fedAvg).to(args.device),n_epochs = args.local_ep)

                    w_fedAvg.append(copy.deepcopy(w_glob_fedAvg))
                    
                    #Weigh models by client dataset size
                    alpha.append(len(dict_users[idx])/len(dataset_train))

            
            # update global model weights    
            w_glob_fedAvg = FedAvg(w_fedAvg, alpha)

            # copy weight to net_glob
            net_glob_fedAvg.load_state_dict(w_glob_fedAvg)

        val_acc_locals, val_acc_mix, val_acc_fedavg, val_acc_e2e, val_acc_3, val_acc_rep, val_acc_repft, val_acc_ft = [], [], [], [], [], [], [], []
        train_acc_ft, train_acc_locals = [], []
        acc_test_l, acc_test_m = [], []
        gate_values = []
        
        for idx in range(args.num_clients):

            client = ClientUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  

            #finetune FedAvg for every client
            print("Finetune %d" %(idx))
            wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(net = copy.deepcopy(net_glob_fedAvg).to(args.device),n_epochs = 200, learning_rate = 1e-5)
            val_acc_ft.append(val_acc_finetuned)
            train_acc_ft.append(train_acc_finetuned)
            
            ft_net = copy.deepcopy(net_glob_fedAvg)
            ft_net.load_state_dict(wt)

            #train local model
            print("Local %d" %(idx))
            w_l, _, val_acc_l, train_acc_l = client.train_finetune(net = copy.deepcopy(net_locals).to(args.device),n_epochs = 200, learning_rate = 1e-4)
            val_acc_locals.append(val_acc_l)
            train_acc_locals.append(train_acc_l)
            
            net_locals_trained = copy.deepcopy(net_locals)
            net_locals_trained.load_state_dict(w_l)

            
            #Train mixture with 2 experts
            print("E2e %d" %(idx))
            w_gate_e2e, _, val_acc_e2e_k, _ = client.train_mix(net_local = copy.deepcopy(net_locals_trained), net_global = copy.deepcopy(net_glob_fedAvg).to(args.device), gate = copy.deepcopy(gates_e2e).to(args.device), train_gate_only=args.train_gate_only, n_epochs = 200, early_stop=True, learning_rate = 1e-5)
                
            val_acc_e2e.append(val_acc_e2e_k)
            
            #Train mixture with 3 experts
            print("Mix3 %d" %(idx))
            nets = [copy.deepcopy(net_locals_trained), copy.deepcopy(net_glob_fedAvg), copy.deepcopy(ft_net)]
            _, _, val_acc_3_k, _ = client.train_3(nets = nets, gate = copy.deepcopy(gates_3).to(args.device), train_gate_only=args.train_gate_only, n_epochs = 200, early_stop=True, learning_rate = 1e-5)
            val_acc_3.append(val_acc_3_k)

            #Train using representations
            #print("Rep %d" %(idx))
            #w_gate_rep, _, val_acc_rep_k, _ = client.train_rep(net_local = copy.deepcopy(net_locals_trained), net_global = copy.deepcopy(net_glob_fedAvg).to(args.device), gate = copy.deepcopy(gate_rep).to(args.device), train_gate_only=args.train_gate_only, n_epochs = 200, early_stop=True)
            #val_acc_rep.append(val_acc_rep_k)
            
            #print("Rep ft %d" %(idx))
            #_, _, val_acc_repft_k, _ = client.train_rep(net_local = copy.deepcopy(ft_net), net_global = copy.deepcopy(net_glob_fedAvg).to(args.device), gate = copy.deepcopy(gate_repft).to(args.device), train_gate_only=args.train_gate_only, n_epochs = 200, early_stop=True)
            #val_acc_repft.append(val_acc_repft_k)
            

            #evaluate FedAvg on local dataset
            val_acc_fed, _ = client.validate(net = net_glob_fedAvg.to(args.device))
            val_acc_fedavg.append(val_acc_fed)


        #Calculate validation and test accuracies
        
        val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)

        train_acc_avg_locals = sum(train_acc_locals)/len(train_acc_locals)
        
        val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
        #val_acc_avg_e2e = np.nan
        
        val_acc_avg_3 = sum(val_acc_3) / len(val_acc_3)
        #val_acc_avg_3 = np.nan
        
        #val_acc_avg_rep = sum(val_acc_rep) / len(val_acc_rep)
        val_acc_avg_rep = np.nan
        
        #val_acc_avg_repft = sum(val_acc_repft) / len(val_acc_repft)
        val_acc_avg_repft = np.nan
        
        val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)
        
        ft_val_acc = sum(val_acc_ft)/len(val_acc_ft)
        ft_train_acc = sum(train_acc_ft)/len(train_acc_ft)
        ft_test_acc = np.nan

        
        with open('save/'+filename,'a') as f1:
            f1.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}'.format(args.dataset, args.model, args.epochs, args.local_ep, args.num_clients, args.iid, args.p, args.opt, args.n_data, args.train_frac, args.train_gate_only, val_acc_avg_e2e, val_acc_avg_locals, val_acc_avg_fedavg, ft_val_acc, val_acc_avg_3, val_acc_avg_rep, val_acc_avg_repft, acc_test_mix, acc_test_locals, acc_test_fedavg, ft_test_acc, ft_train_acc, train_acc_avg_locals, run))
            f1.write("\n")

