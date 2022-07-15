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
import re

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os

class getimage():
    def getInput(path, path1, path2, width, height):
        try:
            img = cv2.imread(path, 1)
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)

            img1 = cv2.imread(path1, 1)
            img1 = cv2.resize(img1, ( width , height ))
            img1 = img1.astype(np.float32)

            img2 = cv2.imread(path2, 1)
            img2 = cv2.resize(img2, ( width , height ))
            img2 = img2.astype(np.float32)

            imgs = np.concatenate((img, img1, img2),axis=2)

            return imgs
        
        except Exception as e:
            print(path, e)

    def getOutput(path, width , height):
        try:
            img = cv2.imread(path, 1)
            img = cv2.resize(img, ( width , height ))
            img = img[:, :, 0]

            label = np.reshape(img, (width*height))
            label = torch.from_numpy(label).long()

            return label
        
        except Exception as e:
            print(path, e)

    def getOutputNorm(path, width, height):
        try:
            img = cv2.imread(path, 1)
            img = cv2.resize(img, ( width , height ))
            img = img[:, :, 0]
            img = img.astype(np.float32)

            if np.max(img) != 0:
                img = img / np.max(img)

            return img
        
        except Exception as e:
            print(path, e)
