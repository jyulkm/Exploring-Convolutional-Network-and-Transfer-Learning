import torch
import argparse
import os, sys, json
from datetime import datetime
from data import get_dataloaders
from engine import *
import matplotlib.pyplot as plt
import numpy as np
import joblib

parser = argparse.ArgumentParser()

parser.add_argument('--log', default=1, type=int,
                    help='Determine if we log the outputs and experiment configurations to local disk')
parser.add_argument('--path', default=datetime.now().strftime('%Y-%m-%d-%H%M%S'), type=str,
                    help='Default log output path if not specified')
parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')    
# Model Related
parser.add_argument('--model', default='vgg', type=str,
                    help='Model being used')
parser.add_argument('--pt_ft', default=1, type=int,
                    help='Determine if the model is for partial fine-tune mode')
parser.add_argument('--model_dir', default=None, type=str,
                    help='Load some saved parameters for the current model')
parser.add_argument('--num_classes', default=20, type=int,
                    help='Number of classes for classification')

# Data Related
parser.add_argument('--bz', default=32, type=int,
                    help='batch size')
parser.add_argument('--shuffle_data', default=True, type=bool,
                    help='Shuffle the data')
parser.add_argument('--normalization_mean', default=(0.485, 0.456, 0.406), type=tuple,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--normalization_std', default=(0.229, 0.224, 0.225), type=tuple,
                    help='Mean value of z-scoring standard deviation for each channel in image')
parser.add_argument('--augmentation', default=0, type=int)

# feel free to add more augmentation/regularization related arguments

# Other Choices & hyperparameters
parser.add_argument('--epoch', default=25, type=int,
                    help='number of epochs')
    # for loss
parser.add_argument('--criterion', default='cross_entropy', type=str,
                    help='which loss function to use')
    # for optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    help='which optimizer to use')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--dampening', default=0, type=float,
                    help='dampening for momentum')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='enables Nesterov momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
# for scheduler
parser.add_argument('--lr_scheduler', default='steplr', type=str,
                    help='learning rate scheduler')
parser.add_argument('--step_size', default=7, type=int,
                    help='Period of learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay.')

# feel free to add more arguments if necessary

args = vars(parser.parse_args())

def main(args):
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)
    model, criterion, optimizer, lr_scheduler = prepare_model(device, args)
    
    #print(np.sum(np.asarray(model.conv1[0].weight[0].T.cpu().detach()) < 0))
    #plt.imsave('test-image', np.asarray(model.conv1[0].weight[0].T.cpu().detach()))
    
    if args['model'] == 'baseline':
        
        train_accuracies = []
        val_accuracies = []
        
        train_losses = []
        val_losses = []
        
        num_epoch = args['epoch']
        for epoch in range(num_epoch):
            train_loss, train_acc, val_loss, val_acc = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders, args)
            _, test_acc, _ = test(model, device, dataloaders[2], criterion)
            print('Current test accuracy: ', test_acc)
            print('Current train accuracy: ', train_acc)
            print('Current val accuracy: ', val_acc)
            print('Current train loss: ', train_loss)
            print('Current val loss: ', val_loss)
            print('Weight values less than 0: ', np.sum(np.asarray(model.conv1[0].weight[0].T.cpu().detach()) < 0))
            print('---------------------------------------------')
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
        _, test_acc, _ = test(model, device, dataloaders[2], criterion)
        
        # Baseline model training and validation accuracy plot
        plt.scatter(np.arange(num_epoch), train_accuracies, color='blue')
        plt.scatter(np.arange(num_epoch), val_accuracies, color='red')
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.title('Baseline Model Accuracy vs. Number of Epochs')
        plt.ylabel('Baseline Model Accuracy')
        plt.xlabel('Number of Epochs')
        plt.savefig('baseline_acc.png')
        
        # Baseline model training and validation loss plot
        plt.scatter(np.arange(num_epoch), train_losses, color='blue')
        plt.scatter(np.arange(num_epoch), val_losses, color='red')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.title('Baseline Model Loss vs. Number of Epochs')
        plt.ylabel('Baseline Model Loss')
        plt.xlabel('Number of Epochs')
        plt.savefig('baseline_loss.png')
        
        print('Baseline model final test accuracy: ', test_acc)

    elif args['model'] == 'custom':
        
        train_accuracies = []
        val_accuracies = []
        
        train_losses = []
        val_losses = []

        num_epoch = args['epoch']
        for epoch in range(num_epoch):
            train_loss, train_acc, val_loss, val_acc = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders, args)
            _, test_acc, _ = test(model, device, dataloaders[2], criterion)
            print('Current test accuracy: ', test_acc)
            print('Current train accuracy: ', train_acc)
            print('Current val accuracy: ', val_acc)
            print('Current train loss: ', train_loss)
            print('Current val loss: ', val_loss)
            print('Weight values less than 0: ', np.sum(np.asarray(model.conv1[0].weight[0].T.cpu().detach()) < 0))
            print('---------------------------------------------')
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
        _, test_acc, _ = test(model, device, dataloaders[2], criterion)
        
        joblib.dump(knn, 'custom.pkl') # saving custom model to pickled file
        
        # Custom model training and validation accuracy plot
        plt.scatter(np.arange(num_epoch), train_accuracies, color='blue')
        plt.scatter(np.arange(num_epoch), val_accuracies, color='red')
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.title('Custom Model Accuracy vs. Number of Epochs')
        plt.ylabel('Custom Model Accuracy')
        plt.xlabel('Number of Epochs')
        plt.savefig('custom_acc.png')
        
        # Custom model training and validation loss plot
        plt.scatter(np.arange(num_epoch), train_losses, color='blue')
        plt.scatter(np.arange(num_epoch), val_losses, color='red')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.title('Custom Model Loss vs. Number of Epochs')
        plt.ylabel('Custom Model Loss')
        plt.xlabel('Number of Epochs')
        plt.savefig('custom_loss.png')
        
        print('Custom model final test accuracy: ', test_acc)
    
    print('Done')

if __name__ == '__main__':
    main(args)
