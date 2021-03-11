import os
import torch_geometric
import torch
import utils
import numpy as np


class Circuit(object):
    #torch_geometric.data.Dataset):
    def __init__(self, description, root, raw_files, transform=None, pre_transform=None):
        #super(Circuit, self).__init__(root, transform, pre_transform)
        self.root = root
        self.description = description
        self.raw_file_names = raw_files
        self.graph_transform = transform
        self.processed_dir = '{root:}/processed/{description:}'.format(root=root, description=description)
        self.data_prefix = "ckt"
        self.data_postfix = "dat"
        self.processed_file_names = []
        
    def process(self):
        if not os.path.isdir(self.processed_dir):
            os.makedirs(self.processed_dir)
        
            for i, path in enumerate(self.raw_file_names):
                #print(path)
                x, edge_index, edge_attr, y = self.graph_transform(path)
                x = torch.FloatTensor(x)
                edge_index = torch.LongTensor(edge_index).t()
                edge_attr = torch.FloatTensor(edge_attr)
                y = torch.FloatTensor(y)

                data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    
                file_name = path.split('/')[-1]
                print(file_name)

                #file_name = "%s_%d.%s" % (self.data_prefix, i, self.data_postfix)
                torch.save(data, os.path.join(self.processed_dir, file_name))
                self.processed_file_names.append(file_name)
        else:
            self.processed_file_names = os.listdir(self.processed_dir)


    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(data_path)
        return data, self.processed_file_names[idx]

    def __getitem__(self, idx):
        #file_name = self.processed_file_names[idx]
        data_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(data_path)
        return data, self.processed_file_names[idx]




if __name__ == '__main__':
    data_root = "../data"
    raw_data_home = "../data/training_data_test"
    processed_data_home = "../data/processed"
    data_files = []
    for (dirpath, dirname, filename) in os.walk(raw_data_home):
        data_files +=  [ os.path.join(dirpath, file) for file in filename ]
    data_files = np.array(data_files)

    np.random.seed(777)
    shuffle = np.random.permutation(len(data_files))

    data_valid = data_files[shuffle[0:20]]
    data_test = data_files[shuffle[20:40]]
    data_train = data_files[shuffle[40:]]

    dataset = Circuit(description='train', root=data_root, raw_files=data_train, transform=utils.parse_dat)
    dataset.process()

    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=3)

    for i, data in enumerate(data_loader):
        print(data)


