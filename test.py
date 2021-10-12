import time
import torch
import numpy as np
import os
import sys
import configs as cfg
from configs import newDir, rmDir
from data_utility import data
from model import myModel
from plot_utility import save_output

class myDataset(torch.utils.data.Dataset):

    def __init__(self, data_src, context = 0):
    
        self.context = context
        self._X = np.pad(np.concatenate(data_src[0],axis=0),((self.context,self.context),(0,0)),'constant',constant_values=0)
        self.X  = self._X

    def __len__(self):
        return len(self.X) - 2*self.context

    def __getitem__(self, index):
        X = self.X[index:index+2*self.context+1].flatten() # flatten
        return torch.from_numpy(X).float()


def test_model(model, test_loader):
    predicted_labels = []
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            #print(len(predicted))
            predicted_labels.append(predicted)

        return predicted_labels

def kagglize_output_labels(predicted_labels,submission_file):
    path = os.getcwd()+f'/{submission_file}.csv'
    fout = open(path,'w')
    fout.write('id,label'+'\n')
    test_labels = torch.cat(predicted_labels,axis=0)
    print("Total # instances : ",len(test_labels))
    for i in range(len(test_labels)):
        fout.write(f'{i},{test_labels[i]}\n')
    fout.close()


if __name__=="__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    data_src = data()   
    model = myModel(cfg.deep6_MLP)
    model.to(device)
    PATH = os.getcwd()+"/state_dict_model_2020.10.03-23:42:29_9.pt"
    model.load_state_dict(torch.load(PATH))
   
    start_time = time.time()
    test_set           = myDataset(data_src.test,cfg.context)
    print("len(test_data)",len(test_set))

    test_loader_args   = dict(shuffle=False, batch_size = cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    test_loader        = torch.utils.data.DataLoader(test_set, **test_loader_args)
    end_time = time.time()
    
    predicted_labels = test_model(model, test_loader)
    save_output(predicted_labels,'predicted_test_labels')
    kagglize_output_labels(predicted_labels,'submission')

