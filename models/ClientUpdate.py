import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch import nn
import torch.nn.functional as F


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientUpdate(object):
    def __init__(self, args, train_set=None,  test_set=None, idxs_train=None, idxs_val=None,idxs_test=None):
        self.args = args
        self.loss_func = nn.NLLLoss()
        self.train_set = DatasetSplit(train_set,idxs_train)
        self.val_set = DatasetSplit(train_set,idxs_val)
        #dataset_length = len(self.train_set)
        #split train into train and val
        #self.train_set, self.val_set = torch.utils.data.random_split(self.train_set,[round(train_frac*dataset_length),round((1-train_frac)*dataset_length)],generator=torch.Generator().manual_seed(23))
        
        self.ldr_train = DataLoader(self.train_set, batch_size=self.args.local_bs, shuffle=True)
        self.ldr_val = DataLoader(self.val_set, batch_size = 1, shuffle=True)
        
        self.test_set = DatasetSplit(test_set,idxs_test)
        self.ldr_test = DataLoader(self.test_set, batch_size = 1, shuffle=True)
        
    def train(self, net, n_epochs,learning_rate):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

        epoch_loss = []

        for iter in range(n_epochs):
            net.train()
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images.float())
                #log_probs = torch.log(probs)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            #val_acc, val_loss = self.validate(net)
            #print(val_acc)
            
        return net.state_dict(), epoch_loss[-1]

    def train_finetune(self, net, n_epochs, learning_rate,val):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        
        patience = 10
        epoch_loss = []
        epoch_train_accuracy = []
        model_best = net.state_dict()
        train_acc_best = np.inf
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        for iter in range(n_epochs):
            net.train()
            batch_loss = []
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images.float())
                #log_probs = torch.log(probs)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs.data, 1)
                correct += (predicted == labels).sum().item()
            train_accuracy = 100.00 * correct / len(self.ldr_train.dataset)
            epoch_train_accuracy.append(train_accuracy)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if(iter%5==0):
                val_acc, val_loss = self.validate(net,val)
                net.train()
                #print(iter, val_loss)
                if(val_loss < val_loss_best - 0.01):
                    counter = 0
                    model_best = net.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    train_acc_best = train_accuracy
                    print("Iter %d | %.2f" %(iter, val_acc_best))
                else:
                    counter = counter + 1

                if counter == patience:
                    return model_best, epoch_loss[-1], val_acc_best, train_acc_best
        
        return model_best, epoch_loss[-1], val_acc_best, train_acc_best
    
    
    
    def train_mix(self, net_local, net_global, gate, train_gate_only, n_epochs, early_stop, learning_rate, val):
        net_local.train()
        net_global.train()
        gate.train()
        
        if(train_gate_only):
            optimizer = torch.optim.Adam(gate.parameters(),lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(list(net_local.parameters()) + 
                                         list(gate.parameters()),lr=learning_rate)

        patience = 10
        epoch_loss = []
        gate_best = gate.state_dict()
        local_best = net_local.state_dict()
        global_best = net_global.state_dict()
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        gate_values_best = 0
        for iter in range(n_epochs):
            
            net_local.train()
            net_global.train()
            gate.train()
            
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net_local.zero_grad()
                net_global.zero_grad()
                gate.zero_grad()
                
                gate_weight = gate(images.float())
                local_probs = net_local(images.float())
                global_probs = net_global(images.float())
                #log_probs = gate_weight[:,0] * local_probs + gate_weight[:,1] * global_probs
                log_probs = gate_weight * local_probs + (1-gate_weight) * global_probs
                loss = self.loss_func(log_probs,labels)
                #loss = torch.mean(-torch.log( gate_weight * torch.exp( -1/2 *(labels - local_probs)**2 ) + (1-gate_weight) * torch.exp( -1/2 *(labels - global_probs)**2) ))
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if(early_stop):
                if(iter%5==0):
                    val_acc, val_loss = self.validate_mix(net_local, net_global, gate, val)
                    net_local.train()
                    net_global.train()
                    gate.train()
                    if(val_loss < val_loss_best - 0.01 ):
                        counter = 0
                        gate_best = gate.state_dict()
                        val_acc_best = val_acc
                        val_loss_best = val_loss
                        
                        gate_best = gate.state_dict()
                        local_best = net_local.state_dict()
                        global_best = net_global.state_dict()
                        
                        print("Iter %d | %.2f" %(iter, val_acc_best))
                    else:
                        counter = counter + 1

                    if counter == patience:
                        return gate_best, local_best, global_best, epoch_loss[-1], val_acc_best
            
        return gate_best, local_best, global_best, epoch_loss[-1], val_acc_best
    
    def train_rep(self, net_local, net_global, gate, train_gate_only, n_epochs, early_stop):
        net_local.train()
        net_global.train()
        gate.train()
        
        if(train_gate_only):
            optimizer = torch.optim.Adam(gate.parameters(),lr=self.lr)
        else:
            optimizer = torch.optim.Adam(list(net_local.parameters()) + list(gate.parameters()) + list(net_global.parameters()),lr=self.lr)

        patience = 10
        epoch_loss = []
        gate_best = gate.state_dict()
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        gate_values_best = 0
        for iter in range(n_epochs):
            
            net_local.train()
            net_global.train()
            gate.train()
            
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net_local.zero_grad()
                net_global.zero_grad()
                gate.zero_grad()

                rep_local, _ = net_local(images)
                rep_global, _ = net_global(images)
                rep = torch.cat((rep_local, rep_global),1)
                log_probs = gate(rep)
                loss = self.loss_func(log_probs,labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if(early_stop):
                if(iter%5==0):
                    val_acc, val_loss = self.validate_rep(net_local, net_global, gate)
                    if(val_loss < val_loss_best ):
                        counter = 0
                        gate_best = gate.state_dict()
                        val_acc_best = val_acc
                        val_loss_best = val_loss
                        print("Iter %d | %.2f" %(iter, val_acc_best))
                    else:
                        counter = counter + 1

                    if counter == patience:
                        return gate_best, epoch_loss[-1], val_acc_best, gate_values_best
            
        return gate_best, epoch_loss[-1], val_acc_best, gate_values_best
    
    
    def train_3(self, nets, gate, train_gate_only, n_epochs, early_stop, learning_rate):
        for i in range(len(nets)):
            nets[i].train()

        gate.train()
        
        if(train_gate_only):
            optimizer = torch.optim.Adam(list(gate.parameters()), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(list(nets[0].parameters()) + list(nets[1].parameters()) + list(gate.parameters()), lr=learning_rate)
            
        patience = 10
        epoch_loss = []
        gate_best = gate.state_dict()
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        #gate_values_best = 0

        
        for iter in range(n_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                for i in range(len(nets)):
                    nets[i].zero_grad()

                gate.zero_grad()
                
                gate_weight = gate(images)
                #gate_weight = gate_weight/torch.sum(gate_weight,axis=1).reshape(-1,1)
                #log_probs = gate_weight * net_local(images) + (1-gate_weight) * net_global(images)
                log_probs = 0
                for i in range(len(nets)):
                    #print(gate_weight.shape)
                    #print(gate_weight[:,i].shape)
                    _, net_probs = nets[i](images)
                    
                    gate_weight_i = gate_weight[:,i].reshape(-1,1)
                    #gate_weight_i = i
                    log_probs += gate_weight_i*net_probs
                    #print(log_probs.shape)
                    
                #log_probs = torch.log(log_probs)
                loss = self.loss_func(log_probs,labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if(early_stop):
                if(iter%5==0):
                    val_acc, val_loss = self.validate_3(nets, gate)
                    if(val_loss < val_loss_best ):
                        counter = 0
                        gate_best = gate.state_dict()
                        val_acc_best = val_acc
                        val_loss_best = val_loss
                        print("Iter %d | %.2f" %(iter, val_acc_best))
                    else:
                        counter = counter + 1

                    if counter == patience:
                        return gate_best, epoch_loss[-1], val_acc_best

        return gate_best, epoch_loss[-1], val_acc_best
    
    def validate(self,net,val):
        if(val):
            dataloader = self.ldr_val
        else:
            dataloader = self.ldr_test
        with torch.no_grad():
            net.eval()
            # validate
            val_loss = 0
            correct = 0
            for idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                log_probs = net(data.float())
                #log_probs = torch.log(probs)
                # sum up batch loss
                val_loss += self.loss_func(log_probs, target).item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            val_loss /= len(dataloader.dataset)
            accuracy = 100.00 * correct / len(dataloader.dataset)
            #print('\nVal set: Average loss: {:.4f} \nAccuracy: {:.2f}%\n'.format(
            #    val_loss, accuracy))
        
        return accuracy.item(), val_loss

    def validate_mix(self, net_l, net_g, gate, val):
        if(val):
            dataloader = self.ldr_val
        else:
            dataloader = self.ldr_test
        with torch.no_grad():
            net_l.eval()
            net_g.eval()
            gate.eval()
            val_loss = 0
            correct = 0
            gate_values = np.array([])
            label_values = np.array([])
            for idx, (data,target) in enumerate(dataloader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                gate_weight = gate(data.float())
                #gate_values = np.append(gate_values,gate_weight.item())
                #label_values = np.append(label_values,target.item())
                
                local_probs = net_l(data.float())
                global_probs = net_g(data.float())
                #log_probs = gate_weight[:,0] * local_probs + gate_weight[:,1] * global_probs
                log_probs = gate_weight * local_probs + (1-gate_weight) * global_probs
                #log_probs = torch.log(log_probs)
                val_loss += self.loss_func(log_probs,target).item()
                #val_loss += torch.mean(-torch.log( gate_weight * torch.exp( -1/2 *(target - local_probs)**2 ) + (1-gate_weight) * torch.exp( -1/2 *(target - global_probs)**2) )).item()
                y_pred = log_probs.data.max(1,keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            val_loss /= len(dataloader.dataset)
            accuracy = 100.00 * correct / len(dataloader.dataset)
            #gl_values = np.concatenate((gate_values.reshape(-1,1), label_values.reshape(-1,1)),axis=1)
            return accuracy.item(), val_loss
        
    def validate_3(self, nets, gate):
        with torch.no_grad():
            for i in range(len(nets)):
                nets[i].eval()
            gate.eval()
            val_loss = 0
            correct = 0
            weights = [1,0]
            #gate_values = np.array([])
            #label_values = np.array([])
            for idx, (data,target) in enumerate(self.ldr_val):
                data, target = data.to(self.args.device), target.to(self.args.device)
                gate_weight = gate(data)
                gate_weight = gate_weight/torch.sum(gate_weight,axis=1).reshape(-1,1)
                log_probs = 0
                for i in range(len(nets)):
                    _, net_probs = nets[i](data)
                    
                    #log_probs += gate_weight[:,i]*net_probs
                    log_probs += weights[i]*net_probs
                    
                #gate_values = np.append(gate_values,gate_weight.item())
                #label_values = np.append(label_values,target.item())
                
                val_loss += self.loss_func(log_probs,target).item()
                y_pred = log_probs.data.max(1,keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            val_loss /= len(self.ldr_val.dataset)
            accuracy = 100.00 * correct / len(self.ldr_val.dataset)
            #gl_values = np.concatenate((gate_values.reshape(-1,1), label_values.reshape(-1,1)),axis=1)
            return accuracy.item(), val_loss
        
    def validate_rep(self, net_l, net_g, gate):
        with torch.no_grad():
            net_l.eval()
            net_g.eval()
            gate.eval()
            val_loss = 0
            correct = 0
            gate_values = np.array([])
            label_values = np.array([])
            for idx, (data,target) in enumerate(self.ldr_val):
                data, target = data.to(self.args.device), target.to(self.args.device)
                
                #gate_values = np.append(gate_values,gate_weight.item())
                #label_values = np.append(label_values,target.item())
                
                rep_local, _ = net_l(data)
                rep_global, _ = net_g(data)
                rep = torch.cat((rep_local, rep_global),1)
                log_probs = gate(rep)
                val_loss += self.loss_func(log_probs,target).item()
                y_pred = log_probs.data.max(1,keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            val_loss /= len(self.ldr_val.dataset)
            accuracy = 100.00 * correct / len(self.ldr_val.dataset)
            gl_values = np.concatenate((gate_values.reshape(-1,1), label_values.reshape(-1,1)),axis=1)
            return accuracy.item(), val_loss
