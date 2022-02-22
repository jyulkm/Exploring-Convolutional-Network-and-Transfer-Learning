# Modified by Colin Wang, Weitang Liu

import model as model_py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np


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
    
    # Getting the appropriate model
    model = get_model(model_type)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Constructing the Adam optimizer
    if model_type == 'baseline':
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Apply the weights to the model and moving model to the device
    model.apply(init_weights)
    model = model.to(device)

    return model, criterion, optimizer, lr_scheduler


def train_model(model, criterion, optimizer, scheduler, device, dataloaders, args=None):
    train_dataloader = dataloaders[0]
    validation_dataloader = dataloaders[1]

    # Put model in train model and train the model using each mini-batch
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # clear the old gradients
        output = model(data)  # compute outputs of the fc layer
        loss = criterion(output, target)
        loss.backward()  # compute gradient for every variables with requires_grad=True
        optimizer.step()  # applied gradient to update the weights

    # Gather the train and validation accuracy and loss
    _, train_acc, train_loss = test(model, device, train_dataloader, criterion)
    _, val_acc, val_loss = test(model, device, validation_dataloader, criterion)

    return train_loss, train_acc, val_loss, val_acc


def test(model, device, test_loader, criterion):

    # Put model in eval mode and test the model on the input data
    model.eval() 
    test_loss = 0
    correct = 0
    with torch.no_grad():  # stop storing gradients for the variables
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Getting the max index of each output to classify images
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += criterion(output, target).item()
        test_loss /= len(test_loader.dataset)

    # Calculate performance
    percent_correct = (100. * correct / len(test_loader.dataset))

    return model, percent_correct, test_loss  # return the model with weight selected by best performance
