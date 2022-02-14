# Modified by Colin Wang, Weitang Liu

def prepare_model(device, args=None):
    # load model, criterion, optimizer, and learning rate scheduler
    
    raise NotImplementedError()

    return model, criterion, optimizer, lr_scheduler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, args=None):
    
    raise NotImplementedError()

    return model # return the model with weight selected by best performance 

# add your own functions if necessary
