import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        # input channel 1, output channel 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # input channel 10, output channel 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # input channel 10, output channel 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3)
        # input channel 10, output channel 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer (fc layer)
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):

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
        # input channel 1, output channel 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # input channel 10, output channel 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # input channel 10, output channel 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3)
        # input channel 10, output channel 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer (fc layer)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(256, 20)
        
    def forward(self, x):
      
        # Forward pass through
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        max_pool = self.max_pool(x3)
        
        x4 = self.conv4(max_pool)
        x5 = self.conv5(x4)        
        x6 = self.conv6(x5)
#         for i in range(128):
#             curr_fm = np.asarray(x6[0][i].cpu().detach())
#             if i+1 < 10:
#                 fn = 'layer6_fm00'+str(i+1)+'.png'
#             elif (i+1>9) & (i+1<100):
#                 fn = 'layer6_fm0'+str(i+1)+'.png'
#             else:
#                 fn = 'layer6_fm'+str(i+1)+'.png'
#             plt.imsave('layer6_fm/'+fn, curr_fm)
        
        avg_pool = self.avg_pool(x6)
        # flatten the cnn features to feed the fully connected layer
        flat = avg_pool.view(-1, 256)
        
        x7 = self.fc1(flat)
        output = self.fc2(x7)
        
        return output


class resnet(nn.Module):
    pass


class vgg(nn.Module):
    pass


def get_model(model_type):
    if model_type == 'baseline':
        return baseline()
    if model_type == 'custom':
        return custom()
    if model_type == 'resnet':
        return resnet()
    if model_type == 'vgg':
        return vgg()
