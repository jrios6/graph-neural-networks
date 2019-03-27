import numpy as np
import torch 
from torch.autograd import Variable
import time
from util.utils import load_data2,load_data3,accuracy

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
    
def load_variables(dataset='cora', cuda=True):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data3(dataset)
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
    
    E = 20000
    V = adj.shape[0]
    if dataset == 'citeseer':
        E = 40000
    elif dataset == 'pubmed':
        E = 110000
    
    E_start = np.zeros((E, V))
    E_end = np.zeros((E, V))
    E_identity = [] # idx of identity edges

    # converting adjacency matrix to edge-to-start, edge-to-end vertex matrix
    count = 0
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j] == 1:
                E_start[count,i] = 1
                E_end[count,j] = 1
                if i == j:
                    E_identity.append(count)
                count += 1
    E_start = E_start[:count]
    E_end = E_end[:count]

    x = Variable(features, requires_grad=False)
    y = Variable(labels)
    E_start = Variable(torch.from_numpy(E_start).float())
    E_end = Variable(torch.from_numpy(E_end).float())
    
    if cuda:
        return x.cuda(), y.cuda(), E_start.cuda(), E_end.cuda(), E_identity, new_edges, idx_train, idx_val, idx_test
    return x, y, E_start, E_end, E_identity, new_edges, idx_train, idx_val, idx_test
    