import argparse
import torch
import numpy as np
import cv2
import itertools
import csv
import pdb
from collections import defaultdict
import time
import glob
import pdb
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

#ファイルインポート
from Model import TrackNet, TrackNet_Focal
from GetImage import getimage

import os

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-weights","--save_weights_path", type = str  )
parser.add_argument('-path','--test_images_path',type=str, default="DataClip/Dataimages/Clip")
parser.add_argument("-output","--output_path", type = str)
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default =640 )
parser.add_argument("--output_height", type=int , default = 360  )
parser.add_argument("--output_width", type=int , default =640 )
parser.add_argument("-peak", "--Ispeak", type=bool, default=False)
parser.add_argument("-tqdm", "--Withtqdm",type=bool,default=False)

args = parser.parse_args()
save_weights_path = args.save_weights_path
n_classes = 256
images_path = args.test_images_path
output_path = args.output_path
input_width =  args.input_width
input_height = args.input_height
output_width =  args.output_width
output_height = args.output_height
withtqdm = args.Withtqdm
IsPeak = args.Ispeak

output_path = "Result/" + output_path + "/Clip"

def getInputFirst(path, width , height):
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)

        return img
    
    except Exception as e:
        print(path, e)

def CatImages(img1, img2):
    img2 = img2.astype("float32")
    im = np.concatenate((img1, img2),axis=2)
    return im

def gaussian_kernel(variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g 

trans = torchvision.transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Net = TrackNet()
Net.to(device)
colors = [ ( i, i, i  ) for i in range(0, 256) ]

Net.load_state_dict(torch.load(save_weights_path))

sigmoid = nn.Sigmoid()

for clip in range(14,22):
    
    images = glob.glob(images_path + str(clip) + "/*.jpg")
    images.sort()

    if not os.path.exists(output_path + str(clip) + "/"):
        os.makedirs(output_path + str(clip) + "/")

    pre_img = np.zeros((output_height, output_width, 3))

    output_name = images[0].replace(images_path, output_path)

    output_img = cv2.resize(pre_img, (1280, 720))

    cv2.imwrite(output_name, output_img)

    if withtqdm:

        with tqdm(total=len(images) - 1, unit="batch") as pbar:

            pbar.set_description("Clip [{}]".format(clip))

            for i in range(1,len(images)):

                output_name = images[i].replace(images_path, output_path)
                X = getimage.getInput(images[i], images[i-1], images[i-2], input_width, input_height)

                X = trans(X)
                X = X.unsqueeze(0)
                X = X.cuda()

                pr = Net(X)

                # FocalLossの場合はこっちを使う
                # pr = Net(X)
                # pr = sigmoid(pr)
                # pr = pr * 255

                pr = pr.to('cpu').detach().numpy().copy()

                pr = pr.reshape((output_height, output_width))
                pr = pr.astype(np.uint8)

                out_img = cv2.resize(pr, (1280, 720))

                cv2.imwrite(output_name, out_img)

                pbar.update(1)
    else:
        print("Clip{}: Running...".format(clip))
        sys.stdout.flush()

        for i in range(2,len(images)):

            output_name = images[i].replace(images_path, output_path)
            X = getimage.getInput(images[i], images[i-1], images[i-2], input_width, input_height)

            X = trans(X)
            X = X.unsqueeze(0)
            X = X.cuda()

            pr = Net(X)

            # FocalLossの場合はこっちを使う
            # pr = Net(X)
            # pr = sigmoid(pr)
            # pr = pr * 255

            pr = pr.to('cpu').detach().numpy().copy()

            pr = pr.reshape((output_height, output_width))
            pr = pr.astype(np.uint8)

            out_img = cv2.resize(pr, (1280, 720))

            cv2.imwrite(output_name, out_img)

        print("Clip{}: Finish".format(clip))
        sys.stdout.flush()
