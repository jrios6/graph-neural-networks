import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb 
import time
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

import sys
import os 
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
CUDA = False

if CUDA and torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(seed)
    
# Helper methods for loading CORA Graph
from utils import load_data3, accuracy

# Load data (GCN)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data3('pubmed')
adj = adj.toarray().astype(float)
adj += np.eye(adj.shape[0])
idx_train = np.argwhere(train_mask).reshape(-1)
idx_val = np.argwhere(val_mask).reshape(-1)
idx_test = np.argwhere(test_mask).reshape(-1)
labels = torch.LongTensor(np.where(labels)[1])

new_edges = []
for v1 in idx_train:
    for v2 in idx_train:
        if v1 != v2 and adj[v1, v2] != 1: # and labels[v1] == labels[v2]:
            new_edges.append((v1,v2))
new_edges = np.array(new_edges)

# For Pudmed
v = adj.shape[0]
Estart = np.zeros((110000, v))
Eend = np.zeros((110000, v))
Eidentity = [] # idx of identity edges

# converting adjacency matrix to edge-to-start, edge-to-end vertex matrix
count = 0
for i in range(v):
    for j in range(v):
        if adj[i,j] == 1:
            Estart[count,i] = 1
            Eend[count,j] = 1
            if i == j:
                Eidentity.append(count)
            count += 1
Estart = Eend[:count]
Eend = Eend[:count]

features_x = Variable(features, requires_grad=False)
train_y = Variable(labels)
E_start = Variable(torch.from_numpy(Estart).float())
E_end = Variable(torch.from_numpy(Eend).float())
E_identity = Eidentity
E_dropin = new_edges

class OurConvNetcell(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_fc=0, dropout_edge=0):
        super(OurConvNetcell, self).__init__()
    
        # conv1
        self.Ui1 = nn.Linear(dim_in, dim_out, bias=False) 
        self.Vi1 = nn.Linear(dim_in, dim_out, bias=False) 
        self.Vj1 = nn.Linear(dim_in, dim_out, bias=False)  
        self.bu1 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        self.bv1 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        
        self.dropout_fc = dropout_fc
        self.dropout_edge = dropout_edge
        
        # conv2
        self.Ui2 = nn.Linear(dim_out, dim_out, bias=False) 
        self.Vi2 = nn.Linear(dim_out, dim_out, bias=False) 
        self.Vj2 = nn.Linear(dim_out, dim_out, bias=False)  
        self.bu2 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        self.bv2 = torch.nn.Parameter( torch.FloatTensor(dim_out), requires_grad=True )
        
        # bn1, bn2
        self.bn1 = torch.nn.BatchNorm1d(dim_out)
        self.bn2 = torch.nn.BatchNorm1d(dim_out)
        
        # resnet
        self.R = nn.Linear(dim_in, dim_out, bias=False) 
        
        # init
        self.init_weights_OurConvNetcell(dim_in, dim_out, 1)
        
         
    def init_weights_OurConvNetcell(self, dim_in, dim_out, gain):   
        # conv1
        scale = gain* np.sqrt( 2.0/ dim_in )
        self.Ui1.weight.data.uniform_(-scale, scale) 
        self.Vi1.weight.data.uniform_(-scale, scale) 
        self.Vj1.weight.data.uniform_(-scale, scale) 
        self.bu1.data.fill_(0)
        self.bv1.data.fill_(0)
        
        # conv2
        scale = gain* np.sqrt( 2.0/ dim_out )
        self.Ui2.weight.data.uniform_(-scale, scale) 
        self.Vi2.weight.data.uniform_(-scale, scale) 
        self.Vj2.weight.data.uniform_(-scale, scale) 
        self.bu2.data.fill_(0)
        self.bv2.data.fill_(0)
        
        # RN
        scale = gain* np.sqrt( 2.0/ dim_in )
        self.R.weight.data.uniform_(-scale, scale)  
        
        
    def forward(self, x, E_start, E_end):
        x = F.dropout(x, self.dropout_fc, training=self.training)
        xin = x
        
        # edge norm
        norm = torch.sum(E_end, 0).reshape(-1,1) #Vx1
        norm = torch.max(norm, torch.ones(norm.shape)) # Vx1

        # conv1
        Uix = self.Ui1(x)  #  V x H_out
        Vix = self.Vi1(x)  #  V x H_out
        Vjx = self.Vj1(x)  #  V x H_out
        x1 = torch.mm(E_end,Vix) + torch.mm(E_start,Vjx) + self.bv1  # E x H_out
        x1 = torch.sigmoid(x1)
        x1 = F.dropout(x1, self.dropout_fc, training=self.training)

        x2 = torch.mm(E_start, Uix)  #  E x H_out
        x = torch.mm(E_end.t(), x1*x2) + self.bu1 #  V x H_out
        
        x = torch.div(x, norm)# norm
        x = self.bn1(x) # bn1
        x = torch.nn.LeakyReLU(0.1)(x) # relu1

        # conv2
        Uix = self.Ui2(x)  #  V x H_out
        Vix = self.Vi2(x)  #  V x H_out
        Vjx = self.Vj2(x)  #  V x H_out
        x1 = torch.mm(E_end,Vix) + torch.mm(E_start,Vjx) + self.bv2  # E x H_out
        x1 = torch.sigmoid(x1)
        x1 = F.dropout(x1, self.dropout_fc, training=self.training)
        
        x2 = torch.mm(E_start, Uix)  #  V x H_out        
        x = torch.mm(E_end.t(), x1*x2) + self.bu2 #  V x H_out
        
        x = torch.div(x, norm) # normalization
        
        x = self.bn2(x) # bn2
        x = x + self.R(xin) # addition
        x = torch.nn.LeakyReLU(0.1)(x) # relu2
        
        return x
        
class Graph_OurConvNet(nn.Module):
    def __init__(self, net_parameters, cora=False, cuda=False):
        super(Graph_OurConvNet, self).__init__()
        
        # parameters
        Voc = net_parameters['Voc']
        D = net_parameters['D']
        nb_clusters_target = net_parameters['nb_clusters_target']
        H = net_parameters['H']
        L = net_parameters['L']
        self.cora = cora
        self.cuda = cuda
        self.dropout_fc = net_parameters['Dropout_fc']
        self.dropout_edge = net_parameters['Dropout_edge']
        self.drop_in = net_parameters['Dropout_in']
        
        # vector of hidden dimensions
        net_layers = []
        for layer in range(L):
            net_layers.append(H)
        
        # CL cells
        # NOTE: Each graph convnet cell uses *TWO* convolutional operations
        net_layers_extended = [net_parameters['features']] + net_layers 
        
        L = len(net_layers)
        list_of_gnn_cells = [] # list of NN cells
        for layer in range(L//2):
            Hin, Hout = net_layers_extended[2*layer], net_layers_extended[2*layer+2]
            list_of_gnn_cells.append(OurConvNetcell(Hin,Hout, self.dropout_fc, self.dropout_edge))
        
        # register the cells for pytorch
        self.gnn_cells = nn.ModuleList(list_of_gnn_cells)
            
        # fc
        Hfinal = net_layers_extended[-1]
        self.fc = nn.Linear(Hfinal,nb_clusters_target) 
        
        # init
        self.init_weights_Graph_OurConvNet(Voc,D,Hfinal,nb_clusters_target,1)
        
        # class variables
        self.D = D
        self.L = L
        self.net_layers_extended = net_layers_extended      
        
        
    def init_weights_Graph_OurConvNet(self, Fin_enc, Fout_enc, Fin_fc, Fout_fc, gain):
        scale = gain* np.sqrt(2.0/ (Fin_fc+Fout_fc))
        self.fc.weight.data.uniform_(-scale, scale)  
        self.fc.bias.data.fill_(0)  
        
    def forward(self, x, E_start, E_end, E_identity, E_dropin):
        E_start = F.dropout(E_start, self.dropout_edge, training=self.training) #FC Dropout

        # convnet cells  
        for layer in range(self.L//2):
            gnn_layer = self.gnn_cells[layer]            
            x = gnn_layer(x,E_start,E_end) # V x H
            
        x = F.dropout(x, self.dropout_fc, training=self.training) #FC Dropout
        x = self.fc(x) # FC
        return x
         
    def loss(self, y, y_target, weight):
        loss = nn.CrossEntropyLoss()(y,y_target)
        return loss
       
    def update(self, lr, l2):
        update = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        return update
    
    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
    def nb_param(self):
        return self.nb_param

def calculate_avg_accuracy(nb_classes, labels, pred_y):
    S = labels.data.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(pred_y).data.cpu().numpy() , axis=1)
    return np.sum(S==C)/S.shape[0]

def update_lr(net, optimizer, average_loss, average_loss_old, lr, decay_rate, early_stopping, verbose):
    # Update LR if > early_stopping and avg val loss is higher
    if average_loss > average_loss_old and lr > early_stopping:
        lr /= decay_rate
        if verbose:
            print('Updating LR to %.7f' % lr)
    return net.update_learning_rate(optimizer, lr), lr

def print_results(iteration, batch_iters, avg_train_acc, running_train_loss, val_accuracy, lr, t_start):
    print('\niteration= %d, train loss(%diter)= %.3f, lr= %.7f, time(%diter)= %.2f' % 
          (iteration, batch_iters, running_train_loss/batch_iters, lr, 
           batch_iters, time.time() - t_start))
    print('val accuracy= %.3f' % (100* val_accuracy))
    print('train accuracy= %.3f' % (100* avg_train_acc))
                         
def train(net, lr, l2, batch_iters, early_stopping, verbose=False):
    ### optimization parameters
    nb_classes = 3
    max_iters = 300
    decay_rate = 1.25
    SAVE_PATH = 'model_state'

    # Optimizer
    optimizer = net.update(lr, l2) 
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = torch.tensor(1e4).cuda() if net.cuda else torch.tensor(1e4)
    best = running_train_acc = running_train_loss = running_val_loss = 0.0
    tab_results = []

    for iteration in range(1, max_iters):  # loop over the dataset multiple times
        # forward, loss
        net.train()
        pred_y = net.forward(features_x, E_start, E_end, E_identity, E_dropin)
        loss = net.loss(pred_y[idx_train], train_y[idx_train], None) 
        train_acc = calculate_avg_accuracy(nb_classes, train_y[idx_train], pred_y[idx_train]) # training acc
        running_train_acc += train_acc    
        running_train_loss += loss.item()

        # backward, update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # validation eval
        net.eval()
        y_eval = net.forward(features_x, E_start, E_end, E_identity, E_dropin)
        val_loss = net.loss(y_eval[idx_val], train_y[idx_val], None) 
        running_val_loss += val_loss.item()

        # learning rate, print results
        if not iteration%batch_iters:
            val_accuracy = calculate_avg_accuracy(nb_classes, train_y[idx_val], y_eval[idx_val])
            average_val_loss = running_val_loss/ batch_iters
            avg_train_acc = running_train_acc/ batch_iters

            # update learning rate 
            if val_accuracy < avg_train_acc and avg_train_acc > 0.8:
                optimizer, lr = update_lr(net, optimizer, average_val_loss, average_loss_old, 
                                          lr, decay_rate, early_stopping, verbose)

            # save intermediate results
            if val_accuracy > best:
                torch.save(net.state_dict(), SAVE_PATH)
                best = val_accuracy
            tab_results.append([iteration,average_val_loss,100* val_accuracy, time.time()-t_start_total])

            if verbose:
                print_results(iteration, batch_iters, avg_train_acc, running_train_loss, val_accuracy, lr, t_start)
            if lr < torch.tensor(early_stopping).cuda() and avg_train_acc - val_accuracy > 0.05:
                print("Early Stopping at %d. Highest Val: %.3f " % (iteration, max([tab_results[i][2] for i in range(len(tab_results))])))
                return max([tab_results[i][2] for i in range(len(tab_results))])
                break

            # reset counters
            t_start = time.time()
            running_train_acc = running_train_loss = running_val_loss = 0.0
            average_loss_old = average_val_loss
    return max([tab_results[i][2] for i in range(len(tab_results))])
                         
CORA = 1
net_parameters = {}
net_parameters['D'] = net_parameters['H'] = 50
net_parameters['features'] = features.shape[1]
net_parameters['Voc'] = 1
net_parameters['nb_clusters_target'] = y_train.shape[1]
                         
TRAIN = True
if TRAIN:
    net_parameters['L'] = 4
    net_parameters['Dropout_fc'] = 0.6
    net_parameters['Dropout_edge'] = 0.4
    net_parameters['Dropout_in'] = 0.0
    net = Graph_OurConvNet(net_parameters, CORA)

    # number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('nb_param=',nb_param,' L=',net_parameters['L'])
    
    lr = 0.001
    l2 = 0.005
    batch_iters = 1
    early_stopping = 5e-5
    train(net, lr, l2, batch_iters, early_stopping, verbose=True)
                         
SAVE_PATH = 'model_state'
nb_classes = 3
net.load_state_dict(torch.load(SAVE_PATH))
net.eval()
# features_x, train_y, E_start, E_end, E_identity, E_dropin = get_cora_dataset(CUDA)
y_eval = net.forward(features_x, E_start, E_end, E_identity, E_dropin)

loss = net.loss(y_eval[idx_test], labels[idx_test], None) 
accuracy = calculate_avg_accuracy(nb_classes, labels[idx_test], y_eval[idx_test])
print('\nloss(100 pre-saved data)= %.3f, accuracy(100 pre-saved data)= %.3f' % (loss.item(), 100* accuracy))
