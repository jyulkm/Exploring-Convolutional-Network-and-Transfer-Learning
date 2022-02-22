import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()

        # input channel 3, output channel 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # input channel 64, output channel 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # input channel 128, output channel 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3)

        # input channel 128, output channel 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 1st fully connected layer; input channel 128, output channel 128
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )

        # 2nd fully connected layer; input channel 128, output channel 20
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        '''
        Making the forward pass through the network. Returns output of 
        the 2nd fully connected layer.
        '''

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        max_pool = self.max_pool(x3)

        x4 = self.conv4(max_pool)

        avg_pool = self.avg_pool(x4)

        # flatten the cnn features to feed the fully connected layer
        flat = avg_pool.view(-1, 128)

        x5 = self.fc1(flat)
        output = self.fc2(x5)

        return output


class custom(nn.Module):
    def __init__(self):
        super(custom, self).__init__()

        # input channel 3, output channel 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # input channel 64, output channel 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # input channel 128, output channel 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3)

        # input channel 128, output channel 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # input channel 256, output channel 256
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # input channel 256, output channel 256
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 1st fully connected layer; input channel 256, output channel 256
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )

        # 2nd fully connected layer; input channel 256, output channel 20
        self.fc2 = nn.Linear(256, 20)
        
    def forward(self, x):
        '''
        Making the forward pass through the network. Returns output of 
        the 2nd fully connected layer.
        '''
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        max_pool = self.max_pool(x3)
        
        x4 = self.conv4(max_pool)
        x5 = self.conv5(x4)        
        x6 = self.conv6(x5)
        
        avg_pool = self.avg_pool(x6)
        
        # flatten the cnn features to feed the fully connected layer
        flat = avg_pool.view(-1, 256)
        
        x7 = self.fc1(flat)
        output = self.fc2(x7)
        
        return output


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model_resnet18 = models.resnet18(pretrained=True)
        
        n_features = self.model_resnet18.fc.in_features
        self.model_resnet18.fc = nn.Linear(n_features, 20)
        print(self.model_resnet18.fc)
        
        freeze_resnet(self.model_resnet18, True) #True: freeze layers, False: unfreeze layers

def freeze_resnet(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.model_vgg16 = models.vgg16(pretrained=True)
        
        num_ftrs = self.model_vgg16.classifier[-1].in_features
        self.model_vgg16.classifier[-1] = nn.Linear(num_ftrs, 20)
        
        set_parameter_requires_grad(self.model_vgg16, False)


                
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            if 'classifier.6' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

def get_model(model_type):
    if model_type == 'baseline':
        return baseline()
    if model_type == 'custom':
        return custom()
    if model_type == 'resnet':
        return resnet()
    if model_type == 'vgg':
        return vgg().model_vgg16