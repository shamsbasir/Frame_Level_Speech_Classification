import os


class newDir:

    def __init__(self,name):
        self.name = name
        if not os.path.exists(name):
            os.mkdir(name)
            print(f"{self.name} directory created! ")
        else :
            print(f"{self.name} directory exists! ")



class rmDir:

    def __init__(self,name):
        self.name = name

        if os.path.exists(name):
            os.rmdir(name)
            print(f"{self.name} directory is removed! ")
        else :
            print(f"{self.name} directory does not exist! ")
#
num_workers = 8
context     = 18
n_epochs    = 6
bins        = 13
input_size  = (2*context+1)*bins
output_size = 346
batch_size  = 510
lr          = 1e-4
weight_decay= 0.0

# model sizes: The best one so far is deep6_MLP
deep1_MLP    = [input_size,2000,1600,1200,800,500,400,output_size]
deep2_MLP    = [input_size,1024,2048,1024,512,512,512,512,output_size]
deep3_MLP    = [input_size,1024,1000,800,600,500,400,output_size]
deep4_MLP    = [input_size,2048,2048,1024,1024,512,512,512,output_size]
deep5_MLP    = [input_size,800,750,730,680,650,600,550,400,output_size]
deep6_MLP    = [input_size,800,750,730,680,640,600,580,540,400,360,output_size] # 10 Layers
deep7_MLP    = [input_size,800,750,730,680,650,620,600,580,550,500,450,400,360,output_size] # 14 Hidden layers
deep8_MLP    = [input_size,900,880,850,810,780,750,700,680,640,600,580,550,510,480,450,400,380,output_size] #15 Hidden layers


