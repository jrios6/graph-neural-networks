import torch
import time
import numpy as np
from model import Graph_OurConvNet
from util.helper import calculate_avg_accuracy, update_lr, print_results, load_variables
import argparse


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--l2', type=float, default=0.005, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50, help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=6, help='Number of conv layers.')
parser.add_argument('--d_conv', type=float, default=0.5, help='Dropout Conv rate (1 - keep probability).')
parser.add_argument('--d_edge', type=float, default=0.5, help='Dropout Edge rate (1 - keep probability).')
parser.add_argument('--d_in', type=float, default=0.0075, help='Dropin rate (1 - keep probability).')
parser.add_argument('--dataset', default="cora", help='Name of Dataset')
parser.add_argument('--save_path', default="model_states/state", help='Name of Dataset')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
if args.cuda:
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(args.seed)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(args.seed)
    
    
def train(net, lr, l2, batch_iters, nb_classes, early_stopping, SAVE_PATH, verbose=False, max_iters = 1000):
    ### optimization parameters
    decay_rate = 1.25

    # Optimizer
    optimizer = net.update(lr, l2) 
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = torch.tensor(1e4).cuda() if net.use_cuda else torch.tensor(1e4)
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
            if val_accuracy < avg_train_acc and avg_train_acc > 0.72:
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


features_x, train_y, E_start, E_end, E_identity, E_dropin, idx_train, idx_val, idx_test = load_variables(args.dataset, args.cuda)

net_parameters = {}
net_parameters['features'] = features_x.shape[1]
net_parameters['nb_clusters_target'] = torch.max(train_y).item()+1
net_parameters['H'] = args.hidden
net_parameters['L'] = args.layers
net_parameters['Dropout_fc'] = args.d_conv
net_parameters['Dropout_edge'] = args.d_edge
net_parameters['Dropout_in'] = args.d_in
lr = args.lr
l2 = args.l2
batch_iters = 20
early_stopping = 50e-6
SAVE_PATH = args.save_path
verbose = True
test_acc = np.zeros((10,1))


for i in range(1):
    net = Graph_OurConvNet(net_parameters, 1, args.cuda)
    
    if args.cuda:
        net.cuda()
        
    print("Training on", args.dataset)
    print("Setting Layers=%d, L2=%.5f, LR=%f, Dropout_FC=%.3f, Dropout_edge=%.3f, Dropin=%.5f" % (net_parameters['L'], l2, lr, net_parameters['Dropout_fc'], net_parameters['Dropout_edge'], net_parameters['Dropout_in']))
    
    train(net, lr, l2, batch_iters, net_parameters['nb_clusters_target'],  early_stopping, SAVE_PATH, verbose, args.epochs)
    
    # Eval
    net.load_state_dict(torch.load(SAVE_PATH))
    net.eval()
    y_eval = net.forward(features_x, E_start, E_end, E_identity, E_dropin)

    loss = net.loss(y_eval[idx_test], train_y[idx_test], None) 
    accuracy = calculate_avg_accuracy(net_parameters['nb_clusters_target'], 
                                      train_y[idx_test], y_eval[idx_test])
    print('\ntest loss = %.3f, test accuracy = %.3f' % (loss.item(), 100* accuracy))
    test_acc[i] += accuracy
