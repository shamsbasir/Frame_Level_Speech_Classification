#@ : Shamsulhaq Basir
from    model            import myModel
from    data_utility     import data_loader,data
import  numpy as np
import  torch
import  sys
import  time
import  matplotlib.pyplot as plt 
import  configs as cfg
from    configs import newDir,rmDir
import datetime
import os
from plot_utility import save_output, visualize
# Modified function from recitation 1
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):  
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        #print(".",end="") 
    end_time = time.time()
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss

# Modified function from recitation 1
def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()

        end_time = time.time()
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss,'Testing Accuracy: ', acc, '%', 'Time :',end_time - start_time)
        return running_loss, acc

def save_model(net,epoch):
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    # Specify a path
    newDir("saved_model")
    PATH = os.getcwd()+f"/saved_model/state_dict_model_{current_time}_{epoch}.pt"
    print("----------- saving the model ....\n") 
    torch.save(net.state_dict(), PATH)



if __name__ =="__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print("cuda found  !")
    data_src     = data()
    train_loader = data_loader(data_src).train
    dev_loader   = data_loader(data_src).dev
    model        = myModel(cfg.deep6_MLP)
    criterion    = torch.nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(),lr =cfg.lr,weight_decay=cfg.weight_decay)
    
    device       = torch.device("cuda" if cuda else "cpu")
    model.to(device)
    print(" --------------- Model --------------")
    print(model)
    print(" ------------ Start of Training  ------------")

    Train_loss   = []
    Test_loss    = []
    Test_acc     = []

    for epoch in range(cfg.n_epochs):
        print(f'epoch: {epoch}')
        train_loss  = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test_model(model, dev_loader, criterion)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)
        print('='*20)
        save_model(model,epoch)
    save_output(Train_loss,'Train_loss')
    save_output(Test_loss,'Test_loss')
    save_output(Test_acc,'Test_acc')
    visualize(Train_loss,'Train_loss')
    visualize(Test_loss,'Test_loss')
    visualize(Test_acc,'Test_acc')
    print("------------ End of Training -----------")
   
