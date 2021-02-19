
import numpy as np
import scipy.sparse as sp
import torch
import os
from enum import Enum, auto
from torch_geometric.data import Data
import shutil



N_NODES_IDX = 0
ADJ_IDX = 1
NODE_FEATURE_DIM = 31
EDGE_FEATURE_DIM = 19
TARGET_DIM = 41

class GATE(Enum):
    AN2D0 = auto()
    AN2D1 = auto()
    AN2D2 = auto()
    AN2D4 = auto()
    AN2D8 = auto()
    AN2XD1 = auto()
    AOI21D0 = auto()
    AOI21D1 = auto()
    AOI21D2 = auto()
    AOI21D4 = auto()
    BUFFD0 = auto()
    BUFFD1 = auto()
    BUFFD2 = auto()
    BUFFD3 = auto()
    BUFFD4 = auto()
    BUFFD6 = auto()
    BUFFD8 = auto()
    BUFFD12 = auto()
    BUFFD16 = auto()
    INVD0 = auto()
    INVD1 = auto()
    INVD2 = auto()
    INVD3 = auto()
    INVD4 = auto()
    INVD6 = auto()
    INVD8 = auto()
    INVD12 = auto()
    INVD16 = auto()
    INVD20 = auto()
    INVD24 = auto()
    MUX2D0 = auto()
    MUX2D1 = auto()
    MUX2D2 = auto()
    MUX2D4 = auto()
    ND2D0 = auto()
    ND2D1 = auto()
    ND2D2 = auto()
    ND2D3 = auto()
    ND2D4 = auto()
    ND2D8 = auto()
    NR2D0 = auto()
    NR2D1 = auto()
    NR2D2 = auto()
    NR2D3 = auto()
    NR2D4 = auto()
    NR2D8 = auto()
    OAI21D0 = auto()
    OAI21D1 = auto()
    OAI21D2 = auto()
    OAI21D4 = auto()
    OR2D0 = auto()
    OR2D1 = auto()
    OR2D2 = auto()
    OR2D4 = auto()
    OR2XD1 = auto()
    XNR2D0 = auto()
    XNR2D1 = auto()
    XNR2D2 = auto()
    XNR2D4 = auto()
    XOR2D0 = auto()
    XOR2D1 = auto()
    XOR2D2 = auto()
    SS = auto()
    ST = auto()


def parse_dat(fname):
    f = open(fname, 'r')

    x = []
    edge_index = []
    edge_attr = []

    N = int(f.readline().split()[0]) + 2

    for i in range(N):
        for j, weight in enumerate(map(int, f.readline().strip().split('\t'))):
            if weight == 0:
                continue
            edge_index.append((i,j))

    # feature coefficient
    x_coef = np.ones(NODE_FEATURE_DIM)
    x_coef[2] = 1e+4
    x_coef[3:] = 1e+12
    

    E = len(edge_index)
    edge_attr = np.zeros((E, EDGE_FEATURE_DIM))
    edge_attr_coef = np.ones(EDGE_FEATURE_DIM)
    edge_attr_coef[4:] = 1e+12

    x = np.zeros((N, NODE_FEATURE_DIM))
    
    # node feature
    for i in range(N):
        if i == 0:
            x[i,0] = getattr(GATE, 'SS').value
        elif i == N-1:
            x[i,0] = getattr(GATE, 'ST').value
        else:
            features = f.readline().strip().split()
            x[i,0] = getattr(GATE, features[0]).value
            x[i,1:] = np.array([ float(val) for val in features[1:] ])


    x = x * x_coef

    #print('x: ', x[1])
    #
    net_features = np.zeros((N, EDGE_FEATURE_DIM))
    for i in range(N-1):
        features = f.readline().strip().split()
        net_features[i] = np.array( [float(val) for val in features[1:]] )
    #net_features = net_features * edge_feature_coef
    
    for i, (n1, n2) in enumerate(edge_index):
        edge_attr[i] = net_features[n1]
    edge_attr = edge_attr * edge_attr_coef

    #print('edge_attr: ', edge_attr[0])

    target = np.zeros(TARGET_DIM)
    for i, val in zip(range(TARGET_DIM), f.readline().strip().split()[1:]):
        target[i] = 1e+12 * float(val)
    #target = [ 1e+12 * float(val) for val in f.readline().strip().split()]
   
    #print('target: ', target)

    f.close()

    
    #target= torch.FloatTensor(target)
    #x = torch.FloatTensor(x)
    #edge_index = torch.LongTensor(edge_index)
    #edge_attr = torch.FloatTensor(edge_attr)

    return (x, edge_index, edge_attr, target)

def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)




'''
def parse_txt(fname="/home/dykim/ConvGNN/data/training_data_test/ckt_10.dat"):

    f = open(fname, 'r')


    N = int(f.readline().split()[0]) + 2
    rows, cols, data = [], [], []
    edges = []


    for i in range(N):
        for j, weight in enumerate(map(int, f.readline().strip().split('\t'))):
            if weight == 0:
                continue
            rows.append(i)
            cols.append(j)
            data.append(weight)
            edges.append([i,j])


    # adj list
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    #del rows, cols, data

    node_feature_dim = 31
    edge_feature_dim = 19

    # feature coefficient
    node_feature_coef = np.ones(node_feature_dim)
    node_feature_coef[2] = 1e+4
    node_feature_coef[3:] = 1e+12
    

    E = len(rows)
    edge_features = np.zeros((E, edge_feature_dim))
    edge_feature_coef = np.ones(edge_feature_dim)
    edge_feature_coef[4:] = 1e+12

    
    node_features = np.zeros((N, node_feature_dim))
    
    # node feature
    for i in range(N):
        if i == 0:
            node_features[i,0] = getattr(GATE, 'SS').value
        elif i == N-1:
            node_features[i,0] = getattr(GATE, 'ST').value
        else:
            features = f.readline().strip().split()
            node_features[i,0] = getattr(GATE, features[0]).value
            node_features[i,1:] = np.array([ float(val) for val in features[1:] ])

    node_features = node_features * node_feature_coef


    #
    net_features = np.zeros((N, edge_feature_dim))
    for i in range(N-1):
        features = f.readline().strip().split()
        net_features[i] = np.array( [float(val) for val in features[1:]] )
    net_features = net_features * edge_feature_coef
    
    for i, (n1, n2) in enumerate(edges):
        edge_features[i] = net_features[n1]

    label_dim = 41
    label = np.zeros(label_dim)
    label = [[ 1e+12 * float(val) for val in f.readline().strip().split()]]

    f.close()

    label = torch.tensor(label)
    edges = torch.tensor(edges, dtype=torch.long)
    node_features = torch.tensor(node_features, dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.long)
    

    data = Data(x=node_features, edge_index=edges, edge_attr=edge_features,y=label)

    print(data)





def load_data(path="/home/dykim/ConvGNN/data/", dataset="ckt"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''
