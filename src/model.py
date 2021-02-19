import torch
import torch.nn as nn
import os
import numpy as np
from utils import parse_dat, save_checkpoint
from torch_geometric.nn import NNConv
from ReadoutFunction import ReadoutFunction
from LogMetric import AverageMeter
from torch.autograd.variable import Variable
from torch.utils.tensorboard import SummaryWriter
import math
import argparse



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
    parser.add_argument('--gpu', default=0, type=int)
    
    args = parser.parse_args()
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



class ConvGNN(nn.Module):
    def __init__(self, args): 
        super(ConvGNN, self).__init__()

        # default args
        self.args = { \
            'aggr': 'add', \
            'type': 'regression',\
            'activation': 'relu' \
        }

        for (key, val) in args.items():
            self.args[key] = val

        self.check_args()
        self.message_func = self.init_message_func()
        self.readout_func = self.init_readout_func()

    def check_args(self):
        required = [ \
            'message_layers', 'readout_layers', 'dropout', \
            'node_feature_size', 'edge_feature_size', 'hidden_state_size', \
            'target_size', 'n_update', 'device' 
        ] 
        optional = ['aggr', 'type' ]
        for key in required:
            if not key in self.args.keys():
                assert "%s is missing (required)" % key

    def init_message_func(self):
   
        dropout = self.args['dropout']
        activation = self.args['activation']
        in_channel = self.args['edge_feature_size']
        num_layers = len(self.args['message_layers'])
        hidden_state_size = self.args['hidden_state_size']
        aggr = self.args['aggr']
        device = self.args['device']

        module_list = []

        for i, num_neuron in enumerate(self.args['message_layers']):
            out_channel = num_neuron
            module_list.append(nn.Linear(in_channel, out_channel))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(p=dropout))
            in_channel = out_channel

            if i == num_layers-1:
                module_list.append(nn.Linear(in_channel, hidden_state_size**2))

        return NNConv(in_channels=hidden_state_size, out_channels=hidden_state_size, nn=nn.Sequential(*module_list), aggr=aggr).to(device)

    def init_readout_func(self):
        dropout = self.args['dropout']
        layers = self.args['readout_layers']
        target_size = self.args['target_size']
        hidden_state_size = self.args['hidden_state_size']
        device = self.args['device']
        return  ReadoutFunction('mpnn', args={'in': hidden_state_size, 'target': target_size, 'dropout':dropout, 'layers':layers, 'device':device})


    def forward(self, x, edge_index, edge_attr):
        # no batch
        hidden_state_size = self.args['hidden_state_size']
        node_feature_size = self.args['node_feature_size']
        device = self.args['device']

        x = torch.FloatTensor(x).to(device)
        edge_index = torch.LongTensor(edge_index).t().to(device)
        edge_attr = torch.FloatTensor(edge_attr).to(device)

        h = []
        h_0 = torch.cat([x, torch.zeros((x.size(0), hidden_state_size - node_feature_size)).to(device)], 1).to(device)
        h.append(h_0.clone())

        for t in range(self.args['n_update']):
            h_t = self.message_func(x=h[t], edge_index=edge_index, edge_attr=edge_attr)
            h.append(h_t)

        res = self.readout_func(h).to(device) 
    
        if self.args['type'] == 'classification':
            res = nn.LogSoftmax()(res)
        return res.squeeze(0)

def custum_loss(output, target, device):
    #output = torch.FloatTensor(output, device=device).unsqueeze(0)
    #target = torch.FloatTensor(target, device=device).unsqueeze(0)
    output = torch.cuda.FloatTensor(output).unsqueeze(0)#.to(device)
    target = torch.cuda.FloatTensor(target).unsqueeze(0)#.to(device)
    data_num = 1000
    param_raw = torch.normal(0, 1, size=(data_num, 10))
    param = torch.zeros(data_num, 10 + NODE_FEATURE_SIZE)
    param[:,0:10] = param_raw[:,0:10]
    cnt = 10
    for k in range(6):
        for m in range(k, 6):
            param[:,cnt] = param_raw[:,k]*param_raw[:,m]
            cnt = cnt + 1

    for k in range(6,10):
        for m in range(k,10):
            param[:,cnt] = param_raw[:,k] * param_raw[:,m]
            cnt = cnt + 1

    A = torch.matmul(param.to(device), output.t().to(device)).to(device)
    B = torch.matmul(param.to(device), target.t().to(device)).to(device)
    loss1 = torch.mean(abs(A-B)).to(device)
    loss2 = torch.mean((output.to(device)-target.to(device))**2).to(device)

    #print('loss1: ', loss1, ' loss2: ', loss2)
    tot_loss = loss1 + loss2
    
    return tot_loss 

def custum_evaluation(output, target, device):
    output = torch.cuda.FloatTensor(output).unsqueeze(0)#.to(device)
    target = torch.cuda.FloatTensor(target).unsqueeze(0)#.to(device)
    return torch.mean(torch.abs(output.to(device)-target.to(device))/torch.abs(target.to(device))).to(device)

def main():
    data_home = "../data/training_data_test"
    data_files = []
    for (dirpath, dirname, filename) in os.walk(data_home):
        data_files +=  [ os.path.join(dirpath, file) for file in filename ]
    data_files = np.array(data_files)
    
    np.random.seed(777)
    shuffle = np.random.permutation(len(data_files))
    
    data_valid = data_files[shuffle[0:20]]
    data_test = data_files[shuffle[20:40]]
    data_train = data_files[shuffle[40:]]

    criterion = custum_loss
    evaluation = custum_evaluation 

    writer = SummaryWriter()

    iargs = get_argument()
    device = get_device(iargs)
    #torch.device('cuda')
    

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
        'opt': iargs.opt,\
        'device': device\
    }

    model = ConvGNN(args)
    #model = model.to(device)

    learning_rate = args['lr'] #1e-04
    opt_type = args['opt']
    best_er1 = 100
    best_loss = math.inf
    best_param = {}
    num_epoch = args['num_epoch']

    optimizer = get_optimizer(model.parameters(), opt_type, learning_rate) 
    #torch.optim.Adam(model.parameters()) #get_optimizer(model.parameters(), args['opt']) #torch.optim.Adam(model.parameters(), learning_rate)

    # model's description
    model_name = "{layer:}_{hidden_state_size:}_{n_update:}_{dropout:}_{num_epoch:}_{lr:}_{opt:}".format(\
            layer='-'.join([str(val) for val in args['message_layers']]),\
            hidden_state_size=args['hidden_state_size'],\
            n_update=args['n_update'],\
            dropout=args['dropout'],\
            num_epoch=args['num_epoch'],\
            lr=args['lr'],\
            opt=args['opt']
            )

    # checkpoint save directory
    save_home = "../save/%s" % model_name
    if not os.path.isdir(save_home):
        os.makedirs(save_home)

    log_file = open('%s/log.txt' % save_home, 'w')

    log_file.write("%s\n" % model_name)
    

    for epoch in range(num_epoch):
        # train
        train_loss, train_err = train(data_train, model, epoch, criterion, evaluation, optimizer, device)
        # validation
        valid_loss, valid_err = validate(data_valid, model, criterion, evaluation, device)

        log_file.write("{epoch:3d}-th Epoch\n".format(epoch=epoch))
        log_file.write(' - Train) Average Error Ratio {err:.3e}; Average Loss {loss:.3e}\n'.format(err=train_err, loss=train_loss))
        log_file.write(' - Valid) Average Error Ratio {err:.3e}; Average Loss {loss:.3e}\n'.format(err=valid_err, loss=valid_loss))

        #is_best = valid_err < best_er1
        is_best = valid_loss < best_loss and not math.isnan(valid_loss)
        if is_best:
            best_loss = min(valid_loss, best_loss) 
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(), }, \
                    is_best=is_best, directory=save_home)

    log_file.write("Best loss : %s\n" % best_loss)
    log_file.close()




def train(data_train, model, epoch, criterion, evaluation, optimizer, device):
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    model.train()

    for i, (x, edge_index, edge_attr, target) in enumerate(map(parse_dat, data_train)):
        output = model(x, edge_index, edge_attr)
        train_loss = criterion(output, target, device)
        train_error = evaluation(output, target, device)

        #print('output: ',output)
        #print('target: ', target)
        #print(train_loss.item())
        losses.update(train_loss.item(), 1)
        error_ratio.update(train_error.item(), 1) #evaluation(output, target).item())

        train_loss.backward()
        optimizer.step()

        '''
        if i % 100 == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4e} ({loss.avg:.4e})\t'
                  'Error Ratio {err.val:.4e} ({err.avg:.4e})'
                  .format(epoch, i, len(data_train), loss=losses, err=error_ratio))
        '''
    return losses.avg, error_ratio.avg


def validate(data_valid, model, criterion, evaluation, device):
    losses = AverageMeter()
    error_ratio = AverageMeter()

    model.eval()
    
    for i, (x, edge_index, edge_attr, target) in enumerate(map(parse_dat, data_valid)):
        output = model(x, edge_index, edge_attr)

        valid_loss = criterion(output, target, device)
        valid_error = evaluation(output, target, device)

        losses.update(valid_loss.item(), 1)
        error_ratio.update(valid_error.item(), 1)


    return losses.avg, error_ratio.avg



if __name__ == '__main__':
    main()
