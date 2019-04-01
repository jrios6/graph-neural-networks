import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def dropin(new_edges, rate, dim=2708, cuda=False):
    np.random.shuffle(new_edges)
    v = new_edges.shape[0]
    E_start = np.zeros((v, dim))
    E_end = np.zeros((v, dim))
    for i in range(0, int(v*rate), 2):
        v1, v2 = new_edges[i]
        E_start[i,v1] = E_end[i,v2] = E_start[i+1,v1] = E_end[i+1,v2] = 1
    E_start = Variable(torch.from_numpy(E_start[:i+2,:]).float())
    E_end = Variable(torch.from_numpy(E_end[:i+2,:]).float())
    
    if cuda:
        return E_start.cuda(), E_end.cuda()
    return E_start, E_end

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
        norm = torch.sum(E_end.t(), 1).reshape(-1,1)
        norm = torch.max(norm, torch.ones(norm.shape).cuda())

        # conv1
        Uix = self.Ui1(x)  #  V x H_out
        Vix = self.Vi1(x)  #  V x H_out
        Vjx = self.Vj1(x)  #  V x H_out
        x1 = torch.mm(E_end,Vix) + torch.mm(E_start,Vjx) + self.bv1  # E x H_out
        x1 = torch.sigmoid(x1)
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
        nb_clusters_target = net_parameters['nb_clusters_target']
        H = net_parameters['H']
        L = net_parameters['L']
        self.use_cuda = cuda
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
        
        self.L = len(net_layers)
        list_of_gnn_cells = [] # list of NN cells
        for layer in range(self.L//2):
            Hin, Hout = net_layers_extended[2*layer], net_layers_extended[2*layer+2]
            list_of_gnn_cells.append(OurConvNetcell(Hin,Hout, self.dropout_fc, self.dropout_edge))
        
        # register the cells for pytorch
        self.gnn_cells = nn.ModuleList(list_of_gnn_cells)
            
        # fc
        Hfinal = net_layers_extended[-1]
        self.fc = nn.Linear(Hfinal,nb_clusters_target) 
        
        # init
        self.init_weights_Graph_OurConvNet(Hfinal,nb_clusters_target,1)
        
    def init_weights_Graph_OurConvNet(self, Fin_fc, Fout_fc, gain):
        scale = gain* np.sqrt(2.0/ (Fin_fc+Fout_fc))
        self.fc.weight.data.uniform_(-scale, scale)  
        self.fc.bias.data.fill_(0)  
        
    def forward(self, x, E_start, E_end, E_identity, E_dropin):
        if self.training:
            # Edge Start+End Dropout for all layers
            num_edges = E_start.shape[0]
            dropout_idx = np.array([i for i in range(num_edges) if i not in E_identity])
            np.random.shuffle(dropout_idx)
            E_start = E_start.clone()
            E_start[dropout_idx[:int(num_edges*self.dropout_edge)]] = 0
            E_end = E_end.clone()
            E_end[dropout_idx[:int(num_edges*self.dropout_edge)]] = 0
            
            # Dropin
            D_start, D_end = dropin(E_dropin, self.drop_in, x.shape[0], self.use_cuda)
            E_start = torch.cat((E_start, D_start), 0)
            E_end = torch.cat((E_end, D_end), 0)
            
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