import sys
import torch
import numpy as np
import cv2
import itertools
import csv
import pdb
from collections import defaultdict
import time
import datetime
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os

#通常のTracknetモデル
class TrackNet(nn.Module):
    def __init__(self):
        super(TrackNet, self).__init__()

        #layer1
        self.conv01 = nn.Conv2d(9, 64, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        #layer2
        self.conv02 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        #layer3
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
        #layer4
        self.conv03 = nn.Conv2d(64, 128, 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        #layer5
        self.conv04 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        #layer6
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
        #layer7
        self.conv05 = nn.Conv2d(128, 256, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        #layer8
        self.conv06 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(256)
        #layer9
        self.conv07 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn7 = nn.BatchNorm2d(256)
        #layyer10
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
        #layer11
        self.conv08 = nn.Conv2d(256, 512, 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(512)
        #layer12
        self.conv09 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn9 = nn.BatchNorm2d(512)
        #layer13
        self.conv10 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn10 = nn.BatchNorm2d(512)
        #layer14
        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest')

        #layer15
        self.deconv01 = nn.Conv2d(512, 256, 3, padding = 1)
        self.bn11 = nn.BatchNorm2d(256)
        #layer16
        self.deconv02 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn12 = nn.BatchNorm2d(256)
        #layer17
        self.deconv03 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn13 = nn.BatchNorm2d(256)
        #layer18
        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')

        #layer19
        self.deconv04 = nn.Conv2d(256, 128, 3, padding = 1)
        self.bn14 = nn.BatchNorm2d(128)
        #layer20
        self.deconv05 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn15 = nn.BatchNorm2d(128)
        #layer21
        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest')

        #layer22
        self.deconv06 = nn.Conv2d(128, 64, 3, padding = 1)
        self.bn16 = nn.BatchNorm2d(64)
        #layer23
        self.deconv07 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn17 = nn.BatchNorm2d(64)
        #layer24
        self.ntrack = nn.Conv2d(64, 256, 3, padding = 1)
        self.bn18 = nn.BatchNorm2d(256)

        #layer25
        self.soft = nn.Softmax(dim=1)

    def forward(self,x):
        #layer1,2,3
        h = F.relu(self.conv01(x))
        h = self.bn1(h)
        h = F.relu(self.conv02(h))
        h = self.bn2(h)
        h, i1 = self.pool1(h)

        #layer4,5,6
        h = F.relu(self.conv03(h))
        h = self.bn3(h)
        h = F.relu(self.conv04(h))
        h = self.bn4(h)
        h, i2 = self.pool2(h)
        #print(h.shape)

        #layer7,8,9,10
        h = F.relu(self.conv05(h))
        h = self.bn5(h)
        h = F.relu(self.conv06(h))
        h = self.bn6(h)
        h = F.relu(self.conv07(h))
        h = self.bn7(h)
        h, i3 = self.pool3(h)
        #print(i3.shape)
        #layer11,12,13,14
        h = F.relu(self.conv08(h))
        h = self.bn8(h)
        h = F.relu(self.conv09(h))
        h = self.bn9(h)
        h = F.relu(self.conv10(h))
        h = self.bn10(h)
        h = self.unpool1(h)

        #layer15,16,17,18
        h = F.relu(self.deconv01(h))
        h = self.bn11(h)
        h = F.relu(self.deconv02(h))
        h = self.bn12(h)
        h = F.relu(self.deconv03(h))
        h = self.bn13(h)
        h = self.unpool2(h)

        #layer19,20,21
        h = F.relu(self.deconv04(h))
        h = self.bn14(h)
        h = F.relu(self.deconv05(h))
        h = self.bn15(h)
        h = self.unpool3(h)

        #layer22,23
        h = F.relu(self.deconv06(h))
        h = self.bn16(h)
        h = F.relu(self.deconv07(h))
        h = self.bn17(h)

        #layer24
        h = F.relu(self.ntrack(h))
        h = self.bn18(h)

        h = h.view(-1,256,360*640)

        return h

#Focallossを導入した際のTracknetのモデル
class TrackNet_Focal(nn.Module):
    def __init__(self):
        super(TrackNet_Focal, self).__init__()

        #layer1
        self.conv01 = nn.Conv2d(9, 64, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        #layer2
        self.conv02 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        #layer3
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
        #layer4
        self.conv03 = nn.Conv2d(64, 128, 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        #layer5
        self.conv04 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        #layer6
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
        #layer7
        self.conv05 = nn.Conv2d(128, 256, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        #layer8
        self.conv06 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(256)
        #layer9
        self.conv07 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn7 = nn.BatchNorm2d(256)
        #layyer10
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
        #layer11
        self.conv08 = nn.Conv2d(256, 512, 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(512)
        #layer12
        self.conv09 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn9 = nn.BatchNorm2d(512)
        #layer13
        self.conv10 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn10 = nn.BatchNorm2d(512)
        #layer14
        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest')

        #layer15
        self.deconv01 = nn.Conv2d(512, 256, 3, padding = 1)
        self.bn11 = nn.BatchNorm2d(256)
        #layer16
        self.deconv02 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn12 = nn.BatchNorm2d(256)
        #layer17
        self.deconv03 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn13 = nn.BatchNorm2d(256)
        #layer18
        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')

        #layer19
        self.deconv04 = nn.Conv2d(256, 128, 3, padding = 1)
        self.bn14 = nn.BatchNorm2d(128)
        #layer20
        self.deconv05 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn15 = nn.BatchNorm2d(128)
        #layer21
        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest')

        #layer22
        self.deconv06 = nn.Conv2d(128, 64, 3, padding = 1)
        self.bn16 = nn.BatchNorm2d(64)
        #layer23
        self.deconv07 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn17 = nn.BatchNorm2d(64)
        #layer24
        self.ntrack = nn.Conv2d(64, 256, 3, padding = 1)
        self.bn18 = nn.BatchNorm2d(256)

        #layer24 * channel 1 *
        self.conv1ch = nn.Conv2d(64, 1, 3, padding = 1)

        #layer25
        self.soft = nn.Softmax(dim=1)

        # Activate
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def forward(self,x):
        #layer1,2,3
        h = self.relu(self.conv01(x))
        h = self.bn1(h)
        h = self.relu(self.conv02(h))
        h = self.bn2(h)
        h, i1 = self.pool1(h)

        #layer4,5,6
        h = self.relu(self.conv03(h))
        h = self.bn3(h)
        h = self.relu(self.conv04(h))
        h = self.bn4(h)
        h, i2 = self.pool2(h)
        #print(h.shape)

        #layer7,8,9,10
        h = self.relu(self.conv05(h))
        h = self.bn5(h)
        h = self.relu(self.conv06(h))
        h = self.bn6(h)
        h = self.relu(self.conv07(h))
        h = self.bn7(h)
        h, i3 = self.pool3(h)
        #print(i3.shape)
        #layer11,12,13,14
        h = self.relu(self.conv08(h))
        h = self.bn8(h)
        h = self.relu(self.conv09(h))
        h = self.bn9(h)
        h = self.relu(self.conv10(h))
        h = self.bn10(h)
        h = self.unpool1(h)

        #layer15,16,17,18
        h = self.relu(self.deconv01(h))
        h = self.bn11(h)
        h = self.relu(self.deconv02(h))
        h = self.bn12(h)
        h = self.relu(self.deconv03(h))
        h = self.bn13(h)
        h = self.unpool2(h)

        #layer19,20,21
        h = self.relu(self.deconv04(h))
        h = self.bn14(h)
        h = self.relu(self.deconv05(h))
        h = self.bn15(h)
        h = self.unpool3(h)

        #layer22,23
        h = self.relu(self.deconv06(h))
        h = self.bn16(h)
        h = self.relu(self.deconv07(h))
        h = self.bn17(h)

        h = self.conv1ch(h)

        return h
