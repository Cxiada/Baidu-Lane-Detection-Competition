import torch
from torch import nn
import numpy as np
# from viz import make_dot
# from graphviz import Digraph
# from models import FcnResNet,Fcnvgg16,FCN8s,FCN8,SegNet,UNet,Unet
from torchvision import models
# from torchsummary import summary
from models.resnet_unet import ResNet34Unet
from models.segnet import SegNet
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

# model = SegNet(3, 2).cuda()
# model = Unet(input_ch=3, output_ch=9).cuda()
model = ResNet34Unet(9).cuda()
# summary(model.cuda(), (3, 640, 320))
# x=torch.randn((1, 3, 640, 320)).cuda()
# re=model(x)
# print(re[0].shape)
# print(model)
