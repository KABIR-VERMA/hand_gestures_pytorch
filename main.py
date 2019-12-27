import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.autograd import Variable
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from Utils import process, getVal
from brain import CNNModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataV, labelV, input_size_val = getVal(device)

def train(model, optimizer):
    model.train()
    loss_sum =0
    acc_sum = 0
    input_size=0
    optimizer.zero_grad()
    for i in range(0,5):
        file = open('./dataset/data'+str(i)+'.pkl', 'rb')
        data= pickle.load(file)
        file.close()
        file = open('./dataset/label'+str(i)+'.pkl', 'rb')
        label= pickle.load(file)
        file.close()
        input_size += len(data)

        n,h,w = data.shape
        data = data.reshape(n,1,h,w)
        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)
        label = label.astype(int)
        label = torch.from_numpy(label)
        data , label = data.to(device), label.to(device)
        data, label = Variable(data), Variable(label)

        for j in range(0,len(data),64):
            bdata = data[j:j+64]
            blabel= label[j:j+64]
            output = model(bdata)
            loss = F.cross_entropy(output, blabel)
            loss_sum += loss.data.item()
            loss.backward()
            optimizer.step()
            predict = output.data.max(1)[1]
            acc = predict.eq(blabel.data).cpu().sum()
            acc_sum +=acc

    return loss_sum/input_size ,acc_sum.item()/input_size 



def evaluate(model):
    model.eval()
    loss_sum =0
    acc_sum = 0
    for j in range(0,input_size_val,64):
        bdata = dataV[j:j+64]
        blabel= labelV[j:j+64]
        output = model(bdata)
        loss = F.cross_entropy(output, blabel)
        loss_sum += loss.data.item()
        predict = output.data.max(1)[1]
        acc = predict.eq(blabel.data).cpu().sum()
        acc_sum +=acc

    return loss_sum/input_size_val ,acc_sum.item()/input_size_val


#initializing  seed  
torch.manual_seed(9372)
# np.random.seed(0)

#Model and optimizer
model = CNNModel()
optimizer = Adam(model.parameters(),0.0001)

#Cpu or Gpu training
model.to(device)
torch.save(model.state_dict(), "./weightedmodel.pth")

best_valid_loss = 1e5
change = 0
strikes =0
status = 'keep_train'

train_losses, val_losses =[],[]
train_accs, val_accs =[],[]
for epoch in range(200):
    print('Epoch', epoch, status)

    train_loss, train_acc = train(model, optimizer)
    print('\t Train loss, accuracy', train_loss, train_acc)
    valid_loss, valid_acc = evaluate(model)
    print('\t Valid loss, best loss, accuracy', valid_loss, best_valid_loss, valid_acc )

    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    train_accs.append(train_acc)
    val_accs.append(valid_acc)

    if valid_loss>best_valid_loss:
        strikes = strikes+1
        if strikes>=8:
            change += 1
            strikes = 0
            print('Current lr change', change)
            if change >=8:
                torch.save(model.state_dict(), "./weightedmodel.pth")
                break 
            else:
                model.load_state_dict(torch.load("./weightedmodel.pth"))
                lr = 0.0001 * np.power(0.1, change)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr  
    else:
        strikes =0
        status = 'keep_train'
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "./weightedmodel.pth")

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend(frameon=True)
plt.show()

plt.clf()

plt.plot(train_accs, label='Training Acc')
plt.plot(val_accs, label='Validation Acc')
plt.legend(frameon=True)
plt.show()

val_loss, val_acc = evaluate(model)
print('Validation loss, accuracy', val_loss, val_acc)


