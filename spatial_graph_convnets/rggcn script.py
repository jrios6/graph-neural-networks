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

seed = 42
np.random.seed(seed)

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(seed)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(seed)
    
# Helper methods for loading CORA Graph
from utils import load_data2,load_data3,accuracy

# Load data (GCN)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data3('cora')
adj = adj.toarray().astype(float)
adj += np.eye(adj.shape[0])
idx_train = np.argwhere(train_mask).reshape(-1)
idx_val = np.argwhere(val_mask).reshape(-1)
idx_test = np.argwhere(test_mask).reshape(-1)
labels = torch.LongTensor(np.where(labels)[1])

# Load data (pyGCN)
# adj, features, labels, idx_train, idx_val, idx_test = load_data2()

# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# adj = adj.toarray()
# adj += np.eye(adj.shape[0])

cora_Estart = np.zeros((20000, 2708))
cora_Eend = np.zeros((20000, 2708))
cora_Eidentity = [] # idx of identity edges

# converting adjacency matrix to edge-to-start, edge-to-end vertex matrix
count = 0
for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        if adj[i,j] == 1:
            cora_Estart[count,i] = 1
            cora_Eend[count,j] = 1
            if i == j:
                cora_Eidentity.append(count)
            count += 1
cora_Estart = cora_Estart[:count]
cora_Eend = cora_Eend[:count]

def get_cora_dataset():
    x = Variable(features, requires_grad=False)
    y = Variable(labels)
    E_start = Variable(torch.from_numpy(cora_Estart).float())
    E_end = Variable(torch.from_numpy(cora_Eend).float())
    
    return x.cuda(), y.cuda(), E_start.cuda(), E_end.cuda(), cora_Eidentity

