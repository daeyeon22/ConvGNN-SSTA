import torch
import torch.nn as nn
import os
import numpy as np
from utils import parse_dat, save_checkpoint
from torch_geometric.nn import NNConv
#from ReadoutFunction import ReadoutFunction
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
from torch_geometric.nn import global_mean_pool, global_add_pool


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
        self.init_message_func()
        self.init_readout_func()

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
   
        #batch = self.args['batch']
        dropout = self.args['dropout']
        activation = self.args['activation']
        num_layers = len(self.args['message_layers'])
        hidden_state_size = self.args['hidden_state_size']
        edge_feature_size = self.args['edge_feature_size']
        node_feature_size = self.args['node_feature_size']
        aggr = self.args['aggr']
        device = self.args['device']

        layers = self.args['message_layers']

        self.state_encoder = nn.Linear(node_feature_size, hidden_state_size).to(device)

        self.message_func = NNConv(in_channels=hidden_state_size, out_channels=hidden_state_size, \
                nn=self.nn(edge_feature_size, hidden_state_size**2, layers, dropout), aggr=aggr).to(device)

        '''
        module_list = []

        for i, num_neuron in enumerate(self.args['message_layers']):
            out_channel = num_neuron
            module_list.append(nn.Linear(in_channel, out_channel))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(p=dropout))
            in_channel = out_channel

            if i == num_layers-1:
                module_list.append(nn.Linear(in_channel, hidden_state_size**2))

        
        #return NNConv(in_channels=hidden_state_size, out_channels=hidden_state_size, nn=nn.Sequential(*module_list), aggr=aggr).to(device)
        '''
    def init_readout_func(self):
        dropout = self.args['dropout']
        layers = self.args['readout_layers']
        target_size = self.args['target_size']
        hidden_state_size = self.args['hidden_state_size']
        device = self.args['device']


        rf1 = self.nn(hidden_state_size*2, target_size, layers, dropout).to(device)
        rf2 = self.nn(hidden_state_size, target_size, layers, dropout).to(device) 
        self.readout_func1 = rf1
        self.readout_func2 = rf2

        #self.readout_func = [\
        #    self.nn(hidden_state_size*2, target_size, layers, dropout).to(device), \
        #    self.nn(hidden_state_size, target_size, layers, dropout).to(device) \
        #]

        
        #return  ReadoutFunction('mpnn', args={'in': hidden_state_size, 'target': target_size, 'dropout':dropout, 'layers':layers, 'device':device})
    
    def nn(self, in_size, out_size, layers, dropout):
        num_layers = len(layers)
        module_list = []
        in_channel = in_size
        for i, num_neuron in enumerate(layers):
            out_channel = num_neuron
            module_list.append(nn.Linear(in_channel, out_channel))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(p=dropout))
            in_channel = out_channel

            if i == num_layers-1:
                module_list.append(nn.Linear(in_channel, out_size))

        return nn.Sequential(*module_list)



    def forward(self, x, edge_index, edge_attr, batch):
        # no batch
        hidden_state_size = self.args['hidden_state_size']
        node_feature_size = self.args['node_feature_size']
        device = self.args['device']

        x = torch.FloatTensor(x).to(device)
        edge_index = torch.LongTensor(edge_index).to(device)
        edge_attr = torch.FloatTensor(edge_attr).to(device)
        batch = torch.LongTensor(batch).to(device)

        
        h = []
        #pd = torch.zeros((x.size(0), hidden_state_size - node_feature_size)).to(device)
        #h_0 = torch.cat( (x, pd), dim=1 ).to(device) 
        #h_0 = torch.cat([x, torch.zeros((x.size(0), hidden_state_size - node_feature_size)).to(device)], 1).to(device)
        
        h_0 = self.state_encoder(x)
        h.append(h_0.clone())

        #h_p = []

        for t in range(self.args['n_update']):
            h_t = h[t]

            #print(type(h[t]), type(edge_index), type(edge_attr))
            #print(h[t].shape, edge_index.shape, edge_attr.shape)
            #print(h.shape)
            h_t = self.message_func(x=h_t, edge_index=edge_index, edge_attr=edge_attr)
            h.append(h_t.clone())
            #print(h_t.shape)
            #tt = global_mean_pool(h_t.to(device), batch)
            #print(tt.shape)
            #h_p.append(tt.clone())            

        #readout
        res = nn.Sigmoid()(self.readout_func1(torch.cat([h[0], h[-1]], 1)))*self.readout_func2(h[-1])
        res = (torch.unsqueeze(torch.sum(h[0],1),1).expand_as(res)>0).type_as(res) * res
        res = global_add_pool(res, batch)
        if self.args['type'] == 'classification':
            res = nn.LogSoftmax()(res)
        return res

        '''
        res = self.readout_func(h, batch).to(device) 
    
        if self.args['type'] == 'classification':
            res = nn.LogSoftmax()(res)
        return res.squeeze(0)
        '''

