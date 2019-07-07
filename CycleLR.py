# LR法寻找学习率

import torch
import numpy as np
import time
import os
import argparse
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from util import label_accuracy_score,label__score
from models.segnet import SegNet
from models.resnet_unet import ResNet34Unet,ResNet34_Unet
from DataReader import TianChiDataset
from logger import Logger
from tqdm import tqdm
from loss import DiscriminativeLoss,FocalLoss,make_one_hot,DiceLoss

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 9  # label类
LEARNING_RATE = 1e-5  # 1e-5
BATCH_SIZE = 2
NUM_EPOCHS = 50  # 100
LOG_INTERVAL = 50
# SIZE = [224, 224]
SIZE = [1280,704]  #W×H

import torchvision.transforms as transforms


def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(train_dataloader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for batch_idx,data in tqdm(enumerate(train_dataloader)):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels,ins_labels = data
        # inputs= torch.autograd.Variable(inputs).cuda()
        img_tensor, sem_tensor = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        ins_tensor = torch.autograd.Variable(ins_labels).cuda()
        optimizer.zero_grad()
        sem_pred,ins_pred = model(img_tensor)
        # loss = criterion_focal(ins_pred, ins_tensor)
        # loss = criterion_focal(outputs, labels)
        loss = criterion_ce(sem_pred, sem_tensor)
        loss +=criterion_disc(ins_pred, ins_tensor, [4] * len(img_tensor))
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


if __name__ == "__main__":
    transformations = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  # 将读取的图片变化
    # logger = Logger('./logs')
    train_path = './train.csv'
    train_dataset = TianChiDataset(train_path, transformations=transformations, size=SIZE)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=16)
    print(len(train_dataloader))
    model = ResNet34Unet(OUTPUT_CHANNELS).cuda()

    criterion_ce = torch.nn.CrossEntropyLoss().cuda()

    # diceloss=DiceLoss()
    # criterion_focal = FocalLoss(9).cuda()
    criterion_disc = DiscriminativeLoss(delta_var=0.1,
                                        delta_dist=0.6,
                                        norm=2,
                                        usegpu=True).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


    logs,losses=find_lr(init_value=1e-6,final_value=10)
    plt.plot(logs[10:-5],losses[10:-5])
    plt.show()
