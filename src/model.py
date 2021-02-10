import torch
import torch.nn as nn
import os
import numpy as np
from utils import parse_dat
from torch_geometric.nn import NNConv
from ReadoutFunction import ReadoutFunction
from LogMetric import AverageMeter
from torch.autograd.variable import Variable

NODE_FEATURE_SIZE = 31
HIDDEN_STATE_SIZE = 95
EDGE_FEATURE_SIZE = 19
MESSAGE_SIZE = 20
TARGET_SIZE = 41
DROPOUT = 0.1
N_LAYERS = 3
LAYERS = [128, 128, 128]
N_UPDATE = 3
CUDA = True

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
            'target_size', 'n_update' 
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

        module_list = []

        for i, num_neuron in enumerate(self.args['message_layers']):
            out_channel = num_neuron
            module_list.append(nn.Linear(in_channel, out_channel))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(p=dropout))
            in_channel = out_channel

            if i == num_layers-1:
                module_list.append(nn.Linear(in_channel, hidden_state_size**2))

        return NNConv(in_channels=hidden_state_size, out_channels=hidden_state_size, nn=nn.Sequential(*module_list), aggr=aggr)

    def init_readout_func(self):
        dropout = self.args['dropout']
        layers = self.args['readout_layers']
        target_size = self.args['target_size']
        hidden_state_size = self.args['hidden_state_size']

        return  ReadoutFunction('mpnn', args={'in': hidden_state_size, 'target': target_size, 'dropout':dropout, 'layers':layers})


    def forward(self, x, edge_index, edge_attr):
        # no batch

        hidden_state_size = self.args['hidden_state_size']
        node_feature_size = self.args['node_feature_size']

        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index).t()
        edge_attr = torch.FloatTensor(edge_attr)

        h = []
        h_0 = torch.cat([x, torch.zeros((x.size(0), hidden_state_size - node_feature_size))], 1) 
        h.append(h_0.clone())

        for t in range(self.args['n_update']):
            h_t = self.message_func(x=h[t], edge_index=edge_index, edge_attr=edge_attr)
            #print('conv forward h_t: ', h_t)
            h.append(h_t)

        #print('h[0]: ', h[0])

        res = self.readout_func(h) 
    
        #print('res: ', res)
        if self.args['type'] == 'classification':
            res = nn.LogSoftmax()(res)
        return res.squeeze(0)

def custum_loss(output, target):

    output = torch.FloatTensor(output)
    target = torch.FloatTensor(target)

    #print('output: ', output)
    #print('target: ', target)
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

    A = torch.matmul(param, output.t())
    B = torch.matmul(param, target.t())

    #print('A: ', A)
    #print('B: ', B)
    loss1 = torch.mean(abs(A-B))
    loss2 = torch.mean((output-target)**2)
    return loss1 + loss2

def custum_evaluation(output, target):
    output = torch.FloatTensor(output)
    target = torch.FloatTensor(target)
    return torch.mean(torch.abs(output-target)/torch.abs(target))

def main():
    data_home = "/home/dykim/ConvGNN/data/training_data_test"
    data_files = []
    for (dirpath, dirname, filename) in os.walk(data_home):
        data_files +=  [ os.path.join(dirpath, file) for file in filename ]
    data_files = np.array(data_files)

    shuffle = np.random.permutation(len(data_files))
    
    data_valid = data_files[shuffle[0:20]]
    data_test = data_files[shuffle[20:40]]
    data_train = data_files[shuffle[40:]]

    criterion = custum_loss
    evaluation = custum_evaluation 

    args = {\
        'message_layers': LAYERS,\
        'readout_layers': LAYERS,\
        'node_feature_size': NODE_FEATURE_SIZE,\
        'hidden_state_size': HIDDEN_STATE_SIZE,\
        'edge_feature_size': EDGE_FEATURE_SIZE,\
        'target_size': TARGET_SIZE,\
        'dropout': DROPOUT,\
        'n_update': N_UPDATE\
    }

    model = ConvGNN(args)
    
    train(data_train, model, 0, criterion, evaluation)

    # train

def train(data_train, model, epoch, criterion, evaluation, _optimizer = 'SGD'):
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    #optimizer = torch.optim.SGD(model.parameters(), 1e-04)
    optimizer = torch.optim.Adam(model.parameters(), 1e-04)

    # for debug
    #for param in model.parameters():
    #    print(type(param), param.size())

    model.train()

    for i, (x, edge_index, edge_attr, target) in enumerate(map(parse_dat, data_train)):
        output = model(x, edge_index, edge_attr)
        train_loss = criterion(output, target)
   
        #print(train_loss.item())
        losses.update(train_loss.item())
        error_ratio.update(evaluation(output, target).item())

        train_loss.backward()
        optimizer.step()


        if i % 100 == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4e} ({loss.avg:.4e})\t'
                  'Error Ratio {err.val:.4e} ({err.avg:.4e})'
                  .format(epoch, i, len(data_train), loss=losses, err=error_ratio))


if __name__ == '__main__':
    main()
