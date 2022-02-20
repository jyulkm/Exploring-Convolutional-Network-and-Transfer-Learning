# Modified by Colin Wang, Weitang Liu

import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import *


def prepare_model(device, args=None):
    learning_rate = args['lr']
    momentum = args['momentum']
    model_type = args['model']
    
    
    def init_weights(m):
        """
        implements xavier intialization (weights)
        """
        if isinstance(m, nn.Conv2d) and model_type == 'baseline':
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            #torch.nn.init.xavier_uniform_(m.bias, gain=1.0)
            torch.nn.init.zeros_(m.bias)

    # load model, criterion, optimizer, and learning rate scheduler
    

    model = get_model(model_type)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    
    if model_type == 'baseline':
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model.apply(init_weights)
    model = model.to(device)

    return model, criterion, optimizer, lr_scheduler


def train_model(model, criterion, optimizer, scheduler, device, dataloaders, args=None):
    train_dataloader = dataloaders[0]
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # clear the old gradients
        output = model(data)  # compute outputs of the fc layer
        loss = criterion(output, target)
        loss.backward()  # compute gradient for every variables with requires_grad=True
        optimizer.step()  # applied gradient to update the weights
        
    return model


def test(model, device, criterion, data_loader, test_data = True):
    model.eval()  # sets model in evaluation (inference) mode. Q3. Why?
    test_loss = 0
    correct = 0
    loss_lst = []
    with torch.no_grad():  # stop storing gradients for the variables
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_lst.append(loss)
            # get the index of maximum fc output. Q4. Why?
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    loss_avg = sum(loss_lst) / len(loss_lst)
    if test_data:
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

    return model, accuracy, loss_avg # return the model with weight selected by best performance
