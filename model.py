import torch
class myModel(torch.nn.Module):
    def __init__(self, size_list):
        super(myModel, self).__init__()
        layers 		   = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(torch.nn.Linear(size_list[i],size_list[i+1]))
            #layers.append(torch.nn.InstanceNorm1d(size_list[i+1]))
            layers.append(torch.nn.BatchNorm1d(size_list[i+1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(size_list[-2], size_list[-1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if  __name__  == "__main__":
    context     = 12
    bins        = 13
    input_size  = (2*context+1)*bins
    output_size = 346
    size_list   = [input_size, 1024, 1024, 512, 512, 256, 128, 64, 32, 16, output_size]
    model       = myModel(size_list)
    criterion   = torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters())
    cuda        = torch.cuda.is_available()
    device      = torch.device("cuda" if cuda else "cpu")
    if device :
        model.to(device)
    print(model)
