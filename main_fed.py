import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
import pandas as pd
from utils.sample_data import mnist_iid, mnist_iid2, mnist_noniid2, cifar_iid, cifar_iid2, cifar_noniid, cifar_noniid2, cifar_dirichlet
from utils.arguments import args_parser
from models.ClientUpdate import ClientUpdate
from models.Models import MLP, CNNCifar, GateCNN, CNNFashion, GateCNNFashion
from models.FederatedAveraging import FedAvg
from models.test_model import test_img, test_img_mix
from torch.utils.tensorboard import SummaryWriter
import os.path
from sys import exit


if __name__ == '__main__':
    filename = 'results_ftmix_fixedglobal_lastfinal'
    filexist = os.path.isfile('save/'+filename) 
    if(not filexist):
        with open('save/'+filename,'a') as f1:
            f1.write('dataset;model;epochs;local_ep;num_clients;iid;p;opt;n_data;frac;lr;train_gate_only;val_acc_avg_e2e;val_acc_avg_e2e_neighbour;val_acc_avg_locals;val_acc_avg_fedavg;ft_val_acc;val_acc_avg_3;val_acc_avg_rep;val_acc_avg_repft;ft_train_acc;train_acc_avg_locals;val_acc_avg_gateonly;localtest_acc_fedavg;localtest_acc_local;localtest_acc_ft;localtest_acc_e2e;test_acc_fedavg;test_acc_local;test_acc_ft;test_acc_e2e;overlap;alpha;freeze;run')

            f1.write('\n')
        
    args=args_parser()
    writer = SummaryWriter(comment=f'lr_{args.lr}_alpha_{args.alpha}_p_{args.p}_opt_{args.opt}')
    for run in range(args.runs):

        args.device = torch.device('cuda:{}'.format(args.gpu))
        train_frac = args.n_data / (args.n_data + args.n_data_val)
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
            if(args.dirichlet):
                dict_users, dict_users_val = cifar_dirichlet(dataset_train, args.num_clients, args.n_data, args.n_data_val, args.alpha)
                dict_users_test = copy.deepcopy(dict_users_val)
            elif args.alpha:
                dict_users, dict_users_val, dict_users_test = cifar_noniid(dataset_train, dataset_test, args.num_clients, args.n_data, args.n_data_val, args.n_data_test, args.alpha)
            else:
                dict_users, dict_users_val, dict_users_test = cifar_noniid2(dataset_train, dataset_test, args.num_clients, args.p, args.n_data, args.n_data_val, args.n_data_test,args.overlap)
                
        elif args.dataset == 'cifar100':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
            if(args.dirichlet):
                dict_users, dict_users_val = cifar_dirichlet(dataset_train, args.num_clients, args.n_data, args.n_data_val, args.alpha)
                dict_users_test = copy.deepcopy(dict_users_val)
            elif args.alpha:
                dict_users, dict_users_val, dict_users_test = cifar_noniid(dataset_train, dataset_test, args.num_clients, args.n_data, args.n_data_val, args.n_data_test, args.alpha)
            else:
                dict_users, dict_users_val, dict_users_test = cifar_noniid2(dataset_train, dataset_test, args.num_clients, args.p, args.n_data, args.n_data_val, args.n_data_test,args.overlap)
                
        elif args.dataset == 'fashion-mnist':
            trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download = True, transform = trans_fashionmnist)
            dataset_test = datasets.FashionMNIST('../data/fashion-mnist',train=False, download = True, transform = trans_fashionmnist)
            
            if args.alpha:
                dict_users, dict_users_val, dict_users_test = cifar_noniid(dataset_train, dataset_test, args.num_clients, args.n_data, args.n_data_val, args.n_data_test, args.alpha)
            else:
                dict_users, dict_users_val, dict_users_test = cifar_noniid2(dataset_train, dataset_test, args.num_clients, args.p, args.n_data, args.n_data_val, args.n_data_test,args.overlap)
        else:
            exit('error: dataset not available')
            
        img_size = dataset_train[0][0].shape

        input_length = 1
        for x in img_size:
            input_length *= x
            
        opt_out = np.random.choice(range(args.num_clients), size = int(args.opt*args.num_clients), replace=False)
        opt_in = [x for x in range(args.num_clients) if x not in opt_out]
        print("opt in")
        print(opt_in)
        #opt[opt_out] = 0  
        
        if (args.model == 'cnn') and (args.dataset in ['cifar10','cifar100']):
            net_glob_fedAvg = CNNCifar(args=args).to(args.device)
            gate_model = GateCNN(args=args).to(args.device)
            net_locals = CNNCifar(args=args).to(args.device)
                
        elif (args.model == 'cnn') and (args.dataset in ['mnist', 'fashion-mnist']):
            net_glob_fedAvg = CNNFashion(args=args).to(args.device)
            gate_model = GateCNNFashion(args=args).to(args.device)
            net_locals = CNNFashion(args=args).to(args.device)

        elif args.model == 'mlp':
            net_glob_fedAvg = MLP(dim_in=input_length, dim_hidden=200, dim_out=args.num_classes).to(args.device)

            #gates = []
            net_locals = []
            gates_e2e = []

            for i in range(args.num_clients):
                #gates.append(GateMLP(dim_in = len_in,dim_hidden=200, dim_out=1).to(args.device))
                gates_e2e.append(GateMLP(dim_in = input_length,dim_hidden=200, dim_out=1).to(args.device))
                net_locals.append(MLP(dim_in=input_length, dim_hidden=200, dim_out=args.num_classes).to(args.device))
                
        else:
            exit('error: no such model')
        
        print(net_glob_fedAvg)

        # training
        val_loss_best = np.inf
        counter = 0
        patience = 5
        for n_iter in range(args.epochs):
            print('Round {:3d}'.format(n_iter))
            
            w_fedAvg = []
            alpha = []
            train_loss = []
            val_loss = []
            val_acc = []
            #m = max(int(args.frac * args.num_clients), 1)
            m = max(int(args.frac), 1)
            idxs_users = np.random.choice(opt_in, m, replace=False) #choose opt-in clients
            #idxs_users=opt_in
            for idx in idxs_users:
                print("FedAvg client %d" %(idx))
                
                client = ClientUpdate(args=args, train_set=dataset_train, test_set = dataset_test, idxs_train=dict_users[idx], idxs_val = dict_users_val[idx], idxs_test = dict_users_test[idx])  
                
                #train FedAvg
                w_glob_fedAvg, train_loss_idx = client.train(net = copy.deepcopy(net_glob_fedAvg).to(args.device),n_epochs = args.local_ep,learning_rate = 5e-5)

                w_fedAvg.append(copy.deepcopy(w_glob_fedAvg))
                train_loss.append(train_loss_idx)
                #Weigh models by client dataset size
                alpha.append(len(dict_users[idx]))

                if(n_iter%40==0):
                    val_acc_fed, val_loss_fed = client.validate(net = net_glob_fedAvg, val=True)
                    val_acc.append(val_acc_fed)
                    val_loss.append(val_loss_fed)

            # update global model weights    
            train_loss_avg = sum(train_loss)/len(train_loss)
            writer.add_scalar('fedAvg_train_loss', train_loss_avg, n_iter)
            if(n_iter%40==0):
                val_loss_avg = sum(val_loss)/len(val_loss)
                val_acc_avg = sum(val_acc)/len(val_acc)
                writer.add_scalar('fedAvg_val_loss', val_loss_avg, n_iter)
                writer.add_scalar('fedAvg_val_acc', val_acc_avg, n_iter)
                if(val_loss_avg < val_loss_best):
                    counter = 0
                    val_loss_best = val_loss_avg
                    w_best_fedavg = w_glob_fedAvg
                else:
                    counter = counter + 1

                if(counter == patience):
                    break

            w_glob_fedAvg = FedAvg(w_fedAvg, alpha)        
            # copy weight to net_glob
            net_glob_fedAvg.load_state_dict(w_glob_fedAvg)
                

        net_glob_fedAvg.load_state_dict(w_best_fedavg)
        
        val_acc_locals, val_acc_mix, val_acc_fedavg, val_acc_e2e, val_acc_3, val_acc_rep, val_acc_repft, val_acc_ft, val_acc_e2e_neighbour, val_acc_gateonly = [], [], [], [], [], [], [], [], [], []
        train_acc_ft, train_acc_locals = [], []
        acc_test_l, acc_test_m = [], []
        gate_values = []

        finetuned = []
        locals_nets = []
        idxs_users = np.random.choice(range(args.num_clients), 20, replace=False) #choose users to evaluate on
        for idx in idxs_users:

            client = ClientUpdate(args=args, train_set=dataset_train, test_set = dataset_test, idxs_train=dict_users[idx], idxs_val = dict_users_val[idx], idxs_test = dict_users_test[idx])  
            
            #finetune FedAvg for every client
            print("Finetune %d" %(idx))
            wt, _, val_acc_finetuned, train_acc_finetuned = client.train_finetune(net = copy.deepcopy(net_glob_fedAvg).to(args.device),n_epochs = 500, learning_rate = 5e-5, val=True)
            val_acc_ft.append(val_acc_finetuned)
            train_acc_ft.append(train_acc_finetuned)
            
            ft_net = copy.deepcopy(net_glob_fedAvg)
            ft_net.load_state_dict(wt)
            finetuned.append(ft_net)
            
            #train local model
            print("Local %d" %(idx))
            net_local_idx = copy.deepcopy(net_locals).to(args.device)
            w_l, _, val_acc_l, train_acc_l = client.train_finetune(net = net_local_idx, n_epochs = 500, learning_rate = 5e-5, val=True)
            
            net_local_idx.load_state_dict(w_l)
            locals_nets.append(net_local_idx)
            
            val_acc_locals.append(val_acc_l)
            train_acc_locals.append(train_acc_l)
        
        mix_local = []
        mix_global = []
        mix_gate = []
        for i, idx in enumerate(idxs_users):
            client = ClientUpdate(args=args, train_set=dataset_train, test_set = dataset_test, idxs_train=dict_users[idx], idxs_val = dict_users_val[idx], idxs_test = dict_users_test[idx])  
            
            print("E2e %d" %(idx))
            gate_idx = copy.deepcopy(gate_model).to(args.device)
            #initialize gate with global model conv parameters
            if(args.freeze):
                gate_idx.conv1 = copy.deepcopy(net_glob_fedAvg.conv1)
                gate_idx.conv2 = copy.deepcopy(net_glob_fedAvg.conv2)
                ct = 0
                for child in gate_idx.children():
                    ct+=1
                    if(ct<4):
                        for param in child.parameters():
                            param.requiers_grad = False
                ct = 0
                #freeze conv layers for finetuned model
                for child in finetuned[i].children():
                    ct+=1
                    if(ct<4):
                        for param in child.parameters():
                            param.requires_grad = False
            
            gate_w, local_w, global_w, _,val_acc_e2e_k = client.train_mix(net_local = copy.deepcopy(finetuned[i]).to(args.device), net_global = copy.deepcopy(net_glob_fedAvg).to(args.device), gate = gate_idx, train_gate_only=args.train_gate_only, n_epochs = 500, early_stop=True, learning_rate = args.lr, val=True)
            
            gate_idx.load_state_dict(gate_w)
            mix_gate.append(copy.deepcopy(gate_idx))
            
            mix_l = copy.deepcopy(net_locals)
            mix_g = copy.deepcopy(net_glob_fedAvg)
            
            mix_l.load_state_dict(local_w)
            mix_g.load_state_dict(global_w)
            
            mix_local.append(mix_l)
            mix_global.append(mix_g)
            
            val_acc_e2e.append(val_acc_e2e_k)
            
            #evaluate FedAvg on local dataset
            val_acc_fed, _ = client.validate(net = net_glob_fedAvg.to(args.device), val=True)
            val_acc_fedavg.append(val_acc_fed)
        if(args.dirichlet):
            localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg = [np.nan],[np.nan],[np.nan],[np.nan]
        else:
            #evaluate with local client test set
            localtest_acc_local, localtest_acc_ft, localtest_acc_e2e, localtest_acc_fedavg = [],[],[],[]
            for i, idx in enumerate(idxs_users):
                client = ClientUpdate(args=args, train_set=dataset_train, test_set = dataset_test, idxs_train=dict_users[idx], idxs_val = dict_users_val[idx], idxs_test = dict_users_test[idx])

                localtest_acc_e2e_idx, _= client.validate_mix(net_l = mix_local[i], 
                                                              net_g = mix_global[i], 
                                                              gate = mix_gate[i], val=False)

                localtest_acc_e2e.append(localtest_acc_e2e_idx)

                localtest_acc_fedavg_idx, _ = client.validate(net = net_glob_fedAvg, val=False)
                localtest_acc_fedavg.append(localtest_acc_fedavg_idx)

                localtest_acc_local_idx, _ = client.validate(net = locals_nets[i], val=False)
                localtest_acc_local.append(localtest_acc_local_idx)

                localtest_acc_ft_idx, _ = client.validate(net = finetuned[i], val=False)
                localtest_acc_ft.append(localtest_acc_ft_idx)

        localtest_acc_local = sum(localtest_acc_local)/len(localtest_acc_local)
        localtest_acc_ft = sum(localtest_acc_ft)/len(localtest_acc_ft)
        localtest_acc_e2e = sum(localtest_acc_e2e)/len(localtest_acc_e2e)
        localtest_acc_fedavg = sum(localtest_acc_fedavg)/len(localtest_acc_fedavg)
            
            
        #evaluate all models on balanced (global) dataset    
        test_acc_local, test_acc_ft, test_acc_e2e = [],[],[]
        print("testing FedAvg...")
        test_acc_fedavg, _ = test_img(net_glob_fedAvg, dataset_test,args)
        for i, idx in enumerate(idxs_users):
            print(idx)
            print("testing Locals...")
            test_acc_local_idx, _ = test_img(locals_nets[i], dataset_test,args)
            print("testing Ft...")
            test_acc_ft_idx, _ = test_img(finetuned[i], dataset_test,args)
            print("testing mixture...")
            test_acc_e2e_idx, _ = test_img_mix(mix_local[i], mix_global[i], mix_gate[i], dataset_test,args)

            test_acc_local.append(test_acc_local_idx)
            test_acc_ft.append(test_acc_ft_idx)
            test_acc_e2e.append(test_acc_e2e_idx)

        test_acc_local = sum(test_acc_local)/len(test_acc_local)
        test_acc_ft = sum(test_acc_ft)/len(test_acc_ft)
        test_acc_e2e = sum(test_acc_e2e)/len(test_acc_e2e)
        
        #Calculate validation and test accuracies
        
        val_acc_avg_locals = sum(val_acc_locals) / len(val_acc_locals)
        #val_acc_avg_locals = np.nan
        
        #train_acc_avg_locals = sum(train_acc_locals)/len(train_acc_locals)
        train_acc_avg_locals = np.nan
        
        val_acc_avg_e2e = sum(val_acc_e2e) / len(val_acc_e2e)
        #val_acc_avg_e2e = np.nan
        
        #val_acc_avg_e2e_neighbour = sum(val_acc_e2e_neighbour) / len(val_acc_e2e_neighbour)
        val_acc_avg_e2e_neighbour = np.nan
        
        #val_acc_avg_3 = sum(val_acc_3) / len(val_acc_3)
        val_acc_avg_3 = np.nan
        
        #val_acc_avg_gateonly = sum(val_acc_gateonly) / len(val_acc_gateonly)
        val_acc_avg_gateonly = np.nan
        
        #val_acc_avg_rep = sum(val_acc_rep) / len(val_acc_rep)
        val_acc_avg_rep = np.nan
        
        #val_acc_avg_repft = sum(val_acc_repft) / len(val_acc_repft)
        val_acc_avg_repft = np.nan
        
        val_acc_avg_fedavg = sum(val_acc_fedavg) / len(val_acc_fedavg)
        
        ft_val_acc = sum(val_acc_ft)/len(val_acc_ft)
        #ft_val_acc = np.nan
        
        ft_train_acc = sum(train_acc_ft)/len(train_acc_ft)
        #ft_train_acc = np.nan

        
        with open('save/'+filename,'a') as f1:
            f1.write(f'{args.dataset};{args.model};{args.epochs};{args.local_ep};{args.num_clients};{args.iid};{args.p};{args.opt};{args.n_data};{args.frac};{args.lr};{args.train_gate_only};{val_acc_avg_e2e};{val_acc_avg_e2e_neighbour};{val_acc_avg_locals};{val_acc_avg_fedavg};{ft_val_acc};{val_acc_avg_3};{val_acc_avg_rep};{val_acc_avg_repft};{ft_train_acc};{train_acc_avg_locals};{val_acc_avg_gateonly};{localtest_acc_fedavg};{localtest_acc_local};{localtest_acc_ft};{localtest_acc_e2e};{test_acc_fedavg};{test_acc_local};{test_acc_ft};{test_acc_e2e};{args.overlap};{args.alpha};{args.freeze};{run}')
            f1.write("\n")

