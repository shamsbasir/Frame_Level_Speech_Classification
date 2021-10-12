import numpy as np
import os
import torch
import configs as cfg
import time
from configs import newDir

class data:
    def __init__(self):
        self.dir          = newDir("data")
        self.source_path  = os.getcwd()+"/data/"
        self.dev_set      = None
        self.train_set    = None
        self.test_set     = None

    
    def data_fetcher(self,name):
        X = np.load(os.path.join(self.source_path , '{}.npy'.format(name)), encoding='bytes',allow_pickle=True)
        Y = None
        if name != "test":
            Y = np.load(os.path.join(self.source_path , '{}_labels.npy'.format(name)), encoding='bytes',allow_pickle=True)
        return X,Y

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = self.data_fetcher('train')
        return self.train_set

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = self.data_fetcher('dev')
        return self.dev_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = self.data_fetcher('test')
        return self.test_set

    

class myDataset(torch.utils.data.Dataset):
    def __init__(self, data_src, context = 0):
        self.context = context
        self._X = np.pad(np.concatenate(data_src[0],axis=0),((self.context,self.context),(0,0)),'constant',constant_values=0)
        self._Y = np.pad(np.concatenate(data_src[1],axis=0),(self.context,self.context),'constant',constant_values=0)
        self.X  = self._X
        self.Y  = self._Y

    def __len__(self):
        return len(self.X) - 2*self.context
  
    def __getitem__(self, index):
        X = self.X[index:index+2*self.context+1].flatten() # flatten
        Y = self.Y[index+self.context]
        return torch.from_numpy(X).float(), torch.from_numpy(np.array(Y))

 


class data_loader:

    def __init__(self,data):
        self.data = data
    
    @property
    def train(self):
    
        start_time = time.time()
        # Training
        train_set           = myDataset(self.data.train,cfg.context)
        train_loader_args   = dict(shuffle=True, batch_size = cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
        train_loader        = torch.utils.data.DataLoader(train_set, **train_loader_args)    
        end_time = time.time()
        print('Time for train_data Loading : ',end_time - start_time, 's')
        
        return train_loader

    @property 
    def dev(self):
        
        # dev loader
        start_time = time.time()
        dev_set             = myDataset(self.data.dev, cfg.context)
        dev_loader_args     = dict(shuffle=False, batch_size = cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
        dev_loader          = torch.utils.data.DataLoader(dev_set, **dev_loader_args)
        end_time = time.time()
        print('Time for dev_data Loading : ', end_time - start_time, 's')
        
        return dev_loader




if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    data_src            = data()
    train_loader        = data_loader(data_src).train
    for batch_index,(item,label) in enumerate(train_loader):
        continue
    print("batch_index : {}, item.shape :{}, label.shape: {}".format(batch_index,item.shape,label.shape))

