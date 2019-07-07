import cv2
import numpy as np
# import os
# import six
# import random
# from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import torch
import pandas as pd
from utils.process_labels import code_init,encode_labels,decode_labels,decode_color_labels,verify_labels
from utils.image_process import crop_resize_data,CLAHE_nomalization,contrast,random_filp
from loss import DiscriminativeLoss
from Baidureader import BaiduDataset
from torch.utils.data import dataloader
import torchvision.transforms as transforms

transformations = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])  # 将读取的图片变化
data=BaiduDataset('./train.csv',transformations=transformations , size=[1024, 384])
image,label,ins=data[0]
print(image.shape,label.shape)

# cv2.imshow('img',image)
# cv2.imshow('label',label)
# cv2.imshow('ins',ins[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# a=list()
# a.append('./170927_063811892_Camera_5.jpg')
# b=list()
# b.append('./170927_063811892_Camera_5_bin.png')
# print(len(a))
# print(len(b))
# c ={'image':a,'label':b}#合并成一个新的字典c
# data = pd.DataFrame(c)
# print(data.head())
# data.to_csv('./train.csv')