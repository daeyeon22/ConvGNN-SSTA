import torch
import torch.nn as nn
import os
import numpy as np
from utils import parse_dat, save_checkpoint
from torch_geometric.nn import NNConv
from LogMetric import AverageMeter
from torch.autograd.variable import Variable
from torch.utils.tensorboard import SummaryWriter
import math
import argparse
#from enum import Enum, auto
import torch_geometric
#from torch_geometric.data import Data
import shutil
from circuit import Circuit
import utils
from torch_geometric.nn import global_mean_pool
from model import ConvGNN

import matplotlib.pyplot as plt
import numbers
from statsmodels.graphics.gofplots import qqplot_2samples
import statsmodels.api as sm


NODE_FEATURE_SIZE = 31
HIDDEN_STATE_SIZE = 95
EDGE_FEATURE_SIZE = 19
MESSAGE_SIZE = 20
TARGET_SIZE = 41
DROPOUT = 0.1
N_LAYERS = 3
LAYERS = [128, 128, 128]
N_UPDATE = 3
NUM_EPOCHS = 100
CUDA = True

def get_argument():
    parser = argparse.ArgumentParser(description='ConvGNN-SSTA')
    parser.add_argument('--hidden_state_size', default=95, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--n_update', default=3, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--opt', default='SparseAdam')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--batch', default=20, type=int)    
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--save', default='../save', type=str)
    iargs = parser.parse_args()

    device = get_device(iargs)
    verbose = iargs.verbose #True
    args = {\
        'message_layers': iargs.layers,\
        'readout_layers': iargs.layers,\
        'node_feature_size': NODE_FEATURE_SIZE,\
        'hidden_state_size': iargs.hidden_state_size,\
        'edge_feature_size': EDGE_FEATURE_SIZE,\
        'target_size': TARGET_SIZE,\
        'dropout': iargs.dropout,\
        'n_update': iargs.n_update,\
        'num_epoch': iargs.n_epochs,\
        'lr': iargs.lr,\
        'batch': iargs.batch,\
        'opt': iargs.opt,\
        'device': device, \
        'verbose': iargs.verbose
    }

    # model's description
    model_name = "{layer:}_{hidden_state_size:}_{n_update:}_{dropout:}_{num_epoch:}_{batch_size:}_{lr:}_{opt:}".format(\
            layer='-'.join([str(val) for val in args['message_layers']]),\
            hidden_state_size=args['hidden_state_size'],\
            n_update=args['n_update'],\
            dropout=args['dropout'],\
            num_epoch=args['num_epoch'],\
            batch_size=args['batch'],\
            lr=args['lr'],\
            opt=args['opt']
            )

    # checkpoint save directory
    save_home = "%s/%s" % (iargs.save, model_name)
    args['model_name'] = model_name
    args['save_home'] = save_home
    args['criterion'] = custum_loss
    args['evaluation'] = custum_evaluation

    return args

def get_optimizer(param, opt_type, lr):
    if opt_type == 'SGD':
        return torch.optim.SGD(param, lr)
    elif opt_type == 'Adam':
        return torch.optim.Adam(param, lr)
    elif opt_type == 'RMSprop':
        return torch.optim.RMSprop(param, lr)
    elif opt_type == 'Nadam':
        return torch.optim.Nadam(param, lr)
    elif opt_type == 'Adamax':
        return torch.optim.Adamax(param, lr)
    elif opt_type == 'Adadelta':
        return torch.optim.Adadelta(param, lr)
    elif opt_type == 'Adagrad':
        return torch.optim.Adagrad(param, lr)
    #elif opt_type == 'SparseAdam':
    #    return torch.optim.SparseAdam(param)
    else:
        return torch.optim.SGD(param, lr)
def get_device(args):
    if args.gpu == None or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
        print("assigned device : ", device)
        return device



def custum_loss(output, target, device):
    target = target.reshape(output.shape).to(device) #view(-1, output.size(1)).to(device)
    data_num = 1000
    param_raw = torch.normal(0, 1, size=(data_num, 10))
    param = torch.zeros(data_num, 10+31)
    param[:,0:10] = param_raw[:,0:10]
    cnt=10
    for k in range(0,6):
        for m in range(k,6):
            param[:,cnt]=param_raw[:,k]*param_raw[:,m];
            cnt = cnt+1

    for k in range(6,10):
        for m in range(k,10):
            param[:,cnt]=param_raw[:,k]*param_raw[:,m];
            cnt = cnt+1
    A=torch.matmul(param.cuda(),output.t().cuda())
    B=torch.matmul(param.cuda(),target.t().cuda())
    loss1 = torch.mean(abs(A - B))
    loss2 = torch.mean((output - target)**2)
    return loss1.cuda()+loss2.cuda()
    #return loss2.cuda()

def custum_evaluation(output, target, device):
    target = target.reshape(output.shape).to(device) #view(-1, output.size(1)).to(device)
    return torch.mean(torch.abs(output.to(device)-target.to(device))/torch.abs(target.to(device))).to(device)


def get_datasets():
    data_root = "../data"
    raw_data_home = "../data/training_data_test"
    processed_data_home = "../data/processed"
    data_files = []
    for (dirpath, dirname, filename) in os.walk(raw_data_home):
        data_files +=  [ os.path.join(dirpath, file) for file in filename ]
    data_files = np.array(data_files)
    shuffle = np.random.permutation(len(data_files))

    raw_data_valid = data_files[shuffle[0:20]]
    raw_data_test = data_files[shuffle[20:40]]
    raw_data_train = data_files[shuffle[40:]]

    train_dataset = Circuit(description='train', root=data_root, raw_files=raw_data_train, transform=utils.parse_dat)
    valid_dataset = Circuit(description='valid', root=data_root, raw_files=raw_data_valid, transform=utils.parse_dat)
    test_dataset = Circuit(description='test', root=data_root, raw_files=raw_data_test, transform=utils.parse_dat)
    train_dataset.process()
    valid_dataset.process()
    test_dataset.process()

    return (train_dataset, valid_dataset, test_dataset)

def get_model(args):
    model = ConvGNN(args)
    num_epoch = args['num_epoch']
    save_home = args['save_home']
    model_name = args['model_name']
    checkpoint_file = os.path.join(save_home, "model_best.pth")
    training = True
    cp_epoch = 0
    cp_train_loss = 0
    cp_valid_loss = 0


    if not os.path.isdir(save_home):
        os.makedirs(save_home)

    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        training = False
        cp_train_loss = checkpoint['train_loss']
        cp_valid_loss = checkpoint['valid_loss']
        cp_epoch = checkpoint['epoch']

    return model, training, cp_epoch, cp_train_loss, cp_valid_loss



def main():
    args = get_argument()
    train_dataset, valid_dataset, test_dataset = get_datasets()

    model_name = args['model_name']
    save_home = args['save_home']
    
    model, training, best_ep, best_train_loss, best_valid_loss = get_model(args)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if training:
        train(model, args, train_dataset, valid_dataset)
        print("Load best model")
        #model, _ = get_model(args)
        model, _, best_ep, best_train_loss, best_valid_loss = get_model(args)
    


    draw_plot(model, args, test_dataset)


    #summary_file = open('%s/summary.txt' % save_home, 'w')
    #summary_file.write('%s epoch %d train_loss %.3f valid_loss %.3f test_loss %.3f\n' % (model_name, best_ep, best_train_loss, best_valid_loss, test_loss))
    #summary_file.close()


    #test(model, args, test_dataset)
    #test(model, args, train_dataset)
    #test(model, args, valid_dataset)


def draw_plot(model, args, test_dataset):

    print("=> start testing")
    batch_size = 1 #args['batch']
    verbose = args['verbose']
    device = args['device']
    save_home = args['save_home']
    model_name = args['model_name']
    criterion = args['criterion']

    fig_home = "%s/figs" % save_home

    if not os.path.isdir(fig_home):
        os.makedirs(fig_home)
    
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    #validation

    xs = np.array([])
    ys = np.array([])

    model.eval()
    for i, data in enumerate(test_loader):
        output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    
        target = data.y
        target = target.reshape(output.shape).to(device) #view(-1, output.size(1)).to(device)
        data_num = 1000
        param_raw = torch.normal(0, 1, size=(data_num, 10))
        param = torch.zeros(data_num, 10+31)
        param[:,0:10] = param_raw[:,0:10]
        cnt=10
        for k in range(0,6):
            for m in range(k,6):
                param[:,cnt]=param_raw[:,k]*param_raw[:,m];
                cnt = cnt+1

        for k in range(6,10):
            for m in range(k,10):
                param[:,cnt]=param_raw[:,k]*param_raw[:,m];
                cnt = cnt+1
        A=torch.matmul(param.to(device),output.t().to(device))
        B=torch.matmul(param.to(device),target.t().to(device))
        
        loss1 = torch.mean(abs(A - B))
        loss2 = torch.mean((output - target)**2)
        
        tot_loss = loss1 + loss2
        A = A.detach().numpy().flatten()
        B = B.detach().numpy().flatten()

        #print(A)
        #print(B)
        print(A.shape)
        print(B.shape)
        print(tot_loss)

        pp_A = sm.ProbPlot(A)
        pp_B = sm.ProbPlot(B)

        print(pp_A)

        fig = qqplot_2samples(pp_A, pp_B, line="45")
        fig.suptitle("loss1 %.2f loss2 %.2f (tot %.2f)" % (loss1, loss2, tot_loss))
        fig.savefig('%s/test_%d.png' % (fig_home, i), dpi=300)
        #plt.show(block=False)
        #plt.scatter(A, B)
        #plt.grid()
        #plt.show(block=False)
        #fig = plt.figure()
        #ax = fig.add_subplot()
        #qqplot(A,B,ax=ax)
        #qqplot(A,B)
        #count_A, bins_count_A = np.histogram(A, bins=100)
        #count_B, bins_count_B = np.histogram(B, bins=100)
        #xs = np.append(xs, A)
        #ys = np.append(ys, B)
        #pdf_A = count_A / np.sum(count_A)
        #print(count_A, bins_count_A)
        #print(count_B, bins_count_B)

    #plt.scatter(xs, ys)
    #plt.grid()
    #plt.show(block=False)
    
    #plt.hist(A)
    #plt.grid()
    #plt.show(block=False)

    #plt.hist(B)
    #plt.grid()
    #plt.show(block=False)

    _ = input("Press enter")

def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
            rug_length=0.05, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.
    Parameters
    ----------
    x, y : array-like
    One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
    Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
    Quantiles to include in the plot. This can be an array of quantiles, in
    which case only the specified quantiles of `x` and `y` will be plotted.
    If this is an int `n`, then the quantiles will be `n` evenly spaced
    points between 0 and 1. If this is None, then `min(len(x), len(y))`
    evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
    Specify the interpolation method used to find quantiles when `quantiles`
    is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
    If True, draw a rug plot representing both samples on the horizontal and
    vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
    Specifies the length of the rug plot lines as a fraction of the total
    vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
    Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
    matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
    Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
    the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
    plt.show(block=False)

if __name__ == '__main__':
    main()
