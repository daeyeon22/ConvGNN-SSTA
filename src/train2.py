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
from plot import draw_plot
from utils import NODE_FEATURE_SIZE, EDGE_FEATURE_SIZE, TARGET_SIZE
from loss import custom_loss2
from statsmodels.graphics.gofplots import qqplot_2samples
import statsmodels.api as sm
import matplotlib.pyplot as plt

#NODE_FEATURE_SIZE = 31
#EDGE_FEATURE_SIZE = 17
#TARGET_SIZE = 41

def get_argument():
    parser = argparse.ArgumentParser(description='ConvGNN-SSTA')
    parser.add_argument('--hidden_state_size', default=95, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', nargs='+', type=int)
    parser.add_argument('--n_update', default=3, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--opt', default='SparseAdam')
    parser.add_argument('--gpu', default=0, type=int)
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
    args['criterion'] = custom_loss2
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

def custum_evaluation(output, target, device):
    target = target.reshape(output.shape).to(device) #view(-1, output.size(1)).to(device)
    return torch.mean(torch.abs(output.to(device)-target.to(device))/torch.abs(target.to(device))).to(device)


def get_datasets():
    data_root = "../data"
    artificial_data_home = "../data/training_data_test"
    iscas_data_home = "../data/iscas"
    processed_data_home = "../data/processed"
    
    artificial_data_files = []
    iscas_data_files = []
    for (dirpath, dirname, filename) in os.walk(artificial_data_home):
        artificial_data_files +=  [ os.path.join(dirpath, file) for file in filename ]
    for (dirpath, dirname, filename) in os.walk(iscas_data_home):
        iscas_data_files +=  [ os.path.join(dirpath, file) for file in filename ]
    
    artificial_data_files = np.array(artificial_data_files)
    iscas_data_files = np.array(iscas_data_files)
    shuffle = np.random.permutation(len(artificial_data_files))

    raw_data_valid = artificial_data_files[shuffle[0:20]]
    raw_data_test = artificial_data_files[shuffle[20:40]]
    raw_data_train = artificial_data_files[shuffle[40:]]
    raw_data_iscas = iscas_data_files

    train_dataset = Circuit(description='train', root=data_root, raw_files=raw_data_train, transform=utils.parse_dat)
    valid_dataset = Circuit(description='valid', root=data_root, raw_files=raw_data_valid, transform=utils.parse_dat)
    test_dataset = Circuit(description='test', root=data_root, raw_files=raw_data_test, transform=utils.parse_dat)
    iscas_dataset = Circuit(description='iscas', root=data_root, raw_files=raw_data_iscas, transform=utils.parse_dat)

    train_dataset.process()
    valid_dataset.process()
    test_dataset.process()
    iscas_dataset.process()

    return (train_dataset, valid_dataset, test_dataset, iscas_dataset)

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
    train_dataset, valid_dataset, test_dataset, iscas_dataset = get_datasets()

    model_name = args['model_name']
    save_home = args['save_home']
    verbose = args['verbose']    
    model, training, best_ep, best_train_loss, best_valid_loss = get_model(args)

    if verbose:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if training:
        train(model, args, train_dataset, valid_dataset)
        print("Load best model")
        #model, _ = get_model(args)
        model, _, best_ep, best_train_loss, best_valid_loss = get_model(args)
    

    test_loss1 = test(model, args, test_dataset)
    test_loss2 = test(model, args, iscas_dataset)

    draw_plot(model, args, test_dataset, dirname='artificial')
    draw_plot(model, args, iscas_dataset, dirname='iscas')

    summary_file = open('%s/summary.txt' % save_home, 'w')
    summary_file.write('%s epoch %d train_loss %.3f valid_loss %.3f test_loss1 %.3f test_loss2 %.3f\n' % (model_name, best_ep, best_train_loss,
                best_valid_loss, test_loss1, test_loss2))
    summary_file.close()

    
def test(model, args, test_dataset):

    print("=> start testing")
    batch_size = args['batch']
    verbose = args['verbose']
    device = args['device']
    save_home = args['save_home']
    model_name = args['model_name']
    criterion = args['criterion']

    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #validation

    test_losses = AverageMeter()

    model.eval()
    for i, (data, _) in enumerate(test_loader):
        output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        
        test_loss = criterion(output, data.y, device)
        test_losses.update(test_loss.item(), data.num_graphs)
        #error_ratio.update(valid_error.item(), data.num_graphs)
       
        if verbose:
            print('[{step:2d}/{tot_step:2d}] Loss {t_loss:.3f}'
                .format(step=i, tot_step=len(test_loader), t_loss=test_loss))


    return test_losses.avg


def train(model, args, train_dataset, valid_dataset):

    print("=> start training")
    writer = SummaryWriter()
    
    verbose = args['verbose']
    device = args['device']
    batch_size = args['batch']
    learning_rate = args['lr'] #1e-04
    opt_type = args['opt']
    num_epoch = args['num_epoch']
    save_home = args['save_home']
    model_name = args['model_name']
    criterion = args['criterion']
    optimizer = get_optimizer(model.parameters(), opt_type, learning_rate) 
    optimizer.zero_grad()


    best_er1 = 100
    best_ep = 0
    best_loss = math.inf
    best_train_loss, best_valid_loss = 0, 0
    best_param = {}
    
    log_file = open('%s/log.txt' % save_home, 'w')
    log_file.write("%s\n" % model_name)

    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch_geometric.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):
        train_losses = AverageMeter()
        valid_losses = AverageMeter()
        
        
        # train
        model.train()
        for i, (data, file_names) in enumerate(train_loader):
            output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            train_loss = criterion(output, data.y, device, file_names)
            train_losses.update(train_loss.item(), data.num_graphs)
            #error_ratio.update(train_error.item(), data.num_graphs) #evaluation(output, target).item())

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()    

        #validation
        model.eval()
        for i, (data, file_names) in enumerate(valid_loader):
            output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            
            valid_loss = criterion(output, data.y, device, file_names)
            valid_losses.update(valid_loss.item(), data.num_graphs)
            #error_ratio.update(valid_error.item(), data.num_graphs)

        train_loss = train_losses.avg
        valid_loss = valid_losses.avg

        if verbose:
            print('[{epoch:2d}/{tot_epoch:2d}] Train Average Loss {t_loss:.3f} Valid Average Loss {v_loss:.3f}'
                .format(epoch=epoch, tot_epoch=num_epoch, t_loss=train_loss, v_loss=valid_loss))

        log_file.write('[{epoch:2d}/{tot_epoch:2d}] Train Average Loss {t_loss:.3f} Valid Average Loss {v_loss:.3f}\n'
                .format(epoch=epoch, tot_epoch=num_epoch, t_loss=train_loss, v_loss=valid_loss))

        #is_best = valid_err < best_er1
        is_best = valid_loss < best_loss and not math.isnan(valid_loss)
        if is_best:
            best_loss = min(valid_loss, best_loss) 
            best_ep = epoch
            best_train_loss = train_loss
            best_valid_loss = best_loss
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(), 'train_loss': best_train_loss,
                    'valid_loss': best_valid_loss, }, \
                    is_best=is_best, directory=save_home)

    if verbose:
        print("(Best) epoch=%d loss=%.3f" % (best_ep, best_loss))

    log_file.write("Best loss : %s\n" % best_loss)
    log_file.close()
    
    #return best_ep, best_train_loss, best_valid_loss

    
def draw_plot2(model, args, test_dataset, dirname=None):
    print("=> draw plot")
    batch_size = 1 #args['batch']
    verbose = args['verbose']
    device = args['device']
    save_home = args['save_home']
    model_name = args['model_name']
    criterion = args['criterion']

    if dirname == None:
        fig_home = "%s/figs" % save_home
    else:
        fig_home = "%s/figs/%s" % (save_home, dirname)

    if not os.path.isdir(fig_home):
        os.makedirs(fig_home)

    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    #validation

    xs = np.array([])
    ys = np.array([])

    model.eval()
    for i, (data, file_names) in enumerate(test_loader):
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

        mu = torch.mean(B, dim=1, keepdim=True).to(device)
        std = torch.std(B, dim=1, keepdim=True).to(device)

        A = (A - mu) / std
        B = (B - mu) / std 

        loss1 = torch.mean((A - B)**2)
        loss2 = torch.mean((output - target)**2)

        tot_loss = loss1 + loss2
        A = A.cpu().detach().numpy().flatten()
        B = B.cpu().detach().numpy().flatten()



        #print(filenames)

        # save figure
        bench_name = file_names[0]
        fig_name = file_names[0].split('.')[0]
        #print(fig_name)
        #print(A)
        #print(B)
        #print(A.shape)
        #print(B.shape)
        #print(tot_loss)

        pp_A = sm.ProbPlot(A)
        pp_B = sm.ProbPlot(B)
        #print(pp_A)

        #print(pp_A.summary())
        #print(pp_B.summary())
        fig = qqplot_2samples(pp_A, pp_B, line="45")
        fig.suptitle("[%s] loss1 %.2f loss2 %.2f (tot %.2f)" % (bench_name, loss1, loss2, tot_loss))
        fig.savefig('%s/%s.png' % (fig_home, fig_name), dpi=150)

        plt.close(fig)


if __name__ == '__main__':
    main()




'''
def main():

    train_dataset, valid_dataset, test_dataset = get_datasets()
    
    criterion = custum_loss
    evaluation = custum_evaluation 

    writer = SummaryWriter()

    iargs = get_argument()
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
        'device': device\
    }

    model = ConvGNN(args)

    batch_size = args['batch']
    learning_rate = args['lr'] #1e-04
    opt_type = args['opt']
    best_er1 = 100
    best_ep = 0
    best_loss = math.inf
    best_train_loss, best_valid_loss = 0, 0
    best_param = {}
    num_epoch = args['num_epoch']

    optimizer = get_optimizer(model.parameters(), opt_type, learning_rate) 

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
    if not os.path.isdir(save_home):
        os.makedirs(save_home)

    log_file = open('%s/log.txt' % save_home, 'w')
    log_file.write("%s\n" % model_name)

    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch_geometric.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):
        # train
        train_loss, train_err = train(train_loader, model, epoch, criterion, evaluation, optimizer, device, verbose)
        # validation
        valid_loss, valid_err = validate(valid_loader, model, criterion, evaluation, device)

        if verbose:
            print('[{epoch:2d}/{tot_epoch:2d}] Train Average Loss {t_loss:.3f} Valid Average Loss {v_loss:.3f}'
                .format(epoch=epoch, tot_epoch=num_epoch, t_loss=train_loss, v_loss=valid_loss))

        log_file.write('[{epoch:2d}/{tot_epoch:2d}] Train Average Loss {t_loss:.3f} Valid Average Loss {v_loss:.3f}\n'
                .format(epoch=epoch, tot_epoch=num_epoch, t_loss=train_loss, v_loss=valid_loss))

        #is_best = valid_err < best_er1
        is_best = valid_loss < best_loss and not math.isnan(valid_loss)
        if is_best:
            best_loss = min(valid_loss, best_loss) 
            best_ep = epoch
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(), }, \
                    is_best=is_best, directory=save_home)

    if verbose:
        print("(Best) epoch=%d loss=%.3f" % (best_ep, best_loss))

    log_file.write("Best loss : %s\n" % best_loss)
    log_file.close()

    summary_file = open('%s/summary.txt' % save_home, 'w')
    summary_file.write('%s epoch %d train_loss %.3f valid_loss %.3f\n' % (model_name, best_ep, best_train_loss, best_valid_loss))
    summary_file.close()


def train(train_loader, model, epoch, criterion, evaluation, optimizer, device, verbose=False):
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    model.train()
    
    for i, data in enumerate(train_loader):
        output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        train_loss = criterion(output, data.y, device)
        train_error = evaluation(output, data.y, device)
        
        losses.update(train_loss.item(), data.num_graphs)
        error_ratio.update(train_error.item(), data.num_graphs) #evaluation(output, target).item())

        train_loss.backward()
        optimizer.step()

        if i % 100 == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4e} ({loss.avg:.4e})\t'
                  'Error Ratio {err.val:.4e} ({err.avg:.4e})'
                  .format(epoch, i, len(data_train), loss=losses, err=error_ratio))
    return losses.avg, error_ratio.avg


def validate(valid_loader, model, criterion, evaluation, device):
    losses = AverageMeter()
    error_ratio = AverageMeter()

    model.eval()
    for i, data in enumerate(valid_loader):
        output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        

        valid_loss = criterion(output, data.y, device)
        valid_error = evaluation(output, data.y, device)

        losses.update(valid_loss.item(), data.num_graphs)
        error_ratio.update(valid_error.item(), data.num_graphs)

    return losses.avg, error_ratio.avg
'''
