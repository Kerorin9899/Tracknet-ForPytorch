import argparse

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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os

#ファイルインポート
from Model import TrackNet, TrackNet_Focal
from GetImage import getimage


seed = 1

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('-trainname','--training_images_name', type = str, default="")
parser.add_argument('-testname', '--testing_images_name', type=str, default=None)
parser.add_argument('-path', "--pathAndlosslistName", type=str)
parser.add_argument('-ih', '--input_height', type=int , default = 360  )
parser.add_argument('-iw','--input_width', type=int , default = 640 )
parser.add_argument('-e','--epochs', type = int, default = 500 )
parser.add_argument('-batch','--batch_size', type = int, default = 2 )
parser.add_argument('-lw','--load_weights', type = str , default = '-1')
parser.add_argument('-se','--step_per_epochs', type = int, default = -1 )
parser.add_argument('-oh', '--output_height', type=int , default = 360  )
parser.add_argument('-ow','--output_width', type=int , default = 640 )
parser.add_argument('--Optim_Param','-optim',type=str,default="Adadelta")
parser.add_argument('--Train_Mode', '-mode', type=str, default="Normal")
parser.add_argument('--Withtqdm', '-tqdm', type=bool, default=True)

dt_now = datetime.datetime.now()

print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))
sys.stdout.flush()

cuda = torch.cuda.is_available()
if cuda:
    print("cuda is available!")

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

args = parser.parse_args()
training_images_name = args.training_images_name
test_images_path = args.testing_images_name
train_batch_size = args.batch_size
savename = args.pathAndlosslistName
n_class = 256
input_height = args.input_height
input_width = args.input_width
output_height = args.output_height
output_width = args.output_width
epochs = args.epochs
load_weights = args.load_weights
step_per_epochs = args.step_per_epochs
opt = args.Optim_Param
train_mode = args.Train_Mode
TestFlag = args.TestMode
withtqdm = args.Withtqdm

n_classes = 256
img_width = 1280
img_height = 720

basename = os.path.basename(__file__)
print(basename)

print("Excecute Program is {}".format(savename))
print("Parameter \nOptimizer : {}\nBatch : {}\nEpoch : {}\nInputSize : {}, {}".format(opt, train_batch_size, epochs, input_width, input_height))
sys.stdout.flush()

if savename is None:
    print("Savename is None")
    sys.exit()

m = nn.Softmax(dim=1)

class Focalloss(nn.Module):
    def __init__(self ,alpha=2, beta=2, gamma=-2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, output, target):
        output = torch.sigmoid(output)
        output = output.clamp(1e-7, 1 - 1e-7)
        loss = _neg_loss(output, target, self.alpha, self.beta)

        return loss

trans = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])

Net = TrackNet()

#optimiserはFocallossの時はAdamを使用すること

if opt == "Adam":
    optimizer = optim.Adam(Net.parameters(), lr=0.001)
elif opt == "Adadelta":
    optimizer = optim.Adadelta(Net.parameters(), lr=1.0)
else:
    print("Optimizer is None!!")
    sys.exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

focal = Focalloss()

CEL = nn.CrossEntropyLoss()
CEL.to(device)

def _neg_loss(pred, gt, alpha, beta):
    ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def criterion(input, target):

    # - Focalloss使用時 - #
    #output = output.unsqueeze(1)
    #loss = focal(input, target)

    #通常のLoss計算
    loss = CEL(input, target)

    return loss

Net.to(device)

if Method == "Tracknet":
    summary(Net, (9, input_height, input_width))
elif Method == "Pre150" or Method == "Pre150Black":
    summary(Net, (7, input_height, input_width))

sys.stdout.flush()

def toTrans(img):
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img

class Mydatasets_Focal(Dataset):
    def __init__(self, transform=None, name=None):
        self.trans = transform

        datapath = []
        datapath1 = []
        datapath2 = []
        labelpath = []
        i = 0
        with open(name) as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                datapath.append(row[0])
                datapath1.append(row[1])
                datapath2.append(row[2])
                labelpath.append(row[3])
                i+=1

        self.data = datapath
        self.data1 = datapath1
        self.data2 = datapath2
        self.label = labelpath

        self.datanum = i

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = getimage.getInput(self.data[idx], self.data1[idx], self.data2[idx], input_width, input_height)

        out_label = getimage.getOutputNorm(self.label[idx], output_width, output_height)

        if self.trans:
            out_data = self.trans(out_data)

        return out_data, out_label

class Mydatasets(Dataset):
    def __init__(self, transform=None, name=None):
        self.trans = transform

        datapath = []
        datapath1 = []
        datapath2 = []
        labelpath = []
        i = 0
        with open(name) as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                datapath.append(row[0])
                datapath1.append(row[1])
                datapath2.append(row[2])
                labelpath.append(row[3])
                i+=1

        self.data = datapath
        self.data1 = datapath1
        self.data2 = datapath2
        self.label = labelpath

        self.datanum = i

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = getimage.getInput(self.data[idx], self.data1[idx], self.data2[idx], input_width, input_height)

        out_label = getimage.getOutput(self.label[idx], output_width, output_height)

        if self.trans:
            out_data = self.trans(out_data)

        return out_data, out_label


trainset = Mydatasets(trans, training_images_name)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size)

if test_images_path is not None:
    testset = Mydatasets(trans, test_images_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

def train():
    Net.train()
    train_loss = 0

    t_start = time.time()

    if withtqdm == True:
        with tqdm(total=len(trainloader), unit="batch") as pbar:

            pbar.set_description("Epoch [{}/{}]".format(ep, epochs))

            for batch_idx, (data, target) in enumerate(trainloader):

                input_tensor =  data.to(device)
                target_tensor =  target.to(device)

                s_tensor = Net(input_tensor)

                optimizer.zero_grad()
                loss = criterion(s_tensor, target_tensor)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix({"loss":loss.item()})
                pbar.update(1)
    else:
        print("Epoch [{}/{}] : Running ...".format(ep, epochs))
        sys.stdout.flush()
        for batch_idx, (data, target) in enumerate(trainloader):

            input_tensor =  data.to(device)
            target_tensor =  target.to(device)

            s_tensor = Net(input_tensor)

            optimizer.zero_grad()
            loss = criterion(s_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print("Epoch [{}/{}] : Finish".format(ep, epochs))
        sys.stdout.flush()
        
    delta = time.time() - t_start

    train_loss /= len(trainloader.dataset)

    print('Epoch [{}/{}], Avg loss: {:.8f}, Time: {:.2f}s'.format(ep,epochs,train_loss, delta))
    sys.stdout.flush()

    return train_loss

def test():
    Net.eval()
    loss = 0
    num = 0

    print("Now Predicting...")
    sys.stdout.flush()
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(testloader):

            input_tensor =  data.to(device)
            target_tensor =  target.to(device)

            s_tensor = Net(input_tensor)

            
time_start = time.time()

if load_weights != "-1":
	Net.load_state_dict(torch.load(load_weights))

loss_list = []
Acc_list = []

Traintime = 0
for ep in range(1, epochs + 1):
    print('Epoch [{}/{}]'.format(ep, epochs))
    prev_loss = float('inf')

    loss = train()

    loss_list.append(loss)

    if ep % 50 == 0:

        torch.save(Net.state_dict(), 'weights/' + str(savename) + '.pth')
        np.save('Losslist/' + str(savename) + '.npy', np.array(loss_list))

torch.save(Net.state_dict(), 'weights/' + str(savename) + '.pth')

print('Train Finished!')

d = time.time() - time_start

dd = 0
dh = 0
dm = 0
ds = d
if d // 60 != 0:
    dm = d // 60
    ds = d % 60

    if dm // 60 != 0:
        dh = dm // 60
        dm = dm % 60

    if dh // 24 != 0:
        dd = dh // 24

print('Total traintime : {:.0f}h{:.0f}m{:.0f}s Total days : {:.0f}days'.format(dh, dm, ds, dd))
