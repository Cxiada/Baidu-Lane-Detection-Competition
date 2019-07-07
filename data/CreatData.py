from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import six
import random
from PIL import Image

from torch.utils import data
import warnings
import matplotlib.pyplot as plt
import torch
import pandas as pd


# --------------------提取全部数据集----------------------------------------
# file_path='/home/cxd/python/pycharm/data/BaiduCar/'
#
#
# label_dirname = os.path.join(file_path, 'Label_road02/Label/')
# labels = list()
# for dire in os.listdir(label_dirname):
#     for dir in os.listdir(os.path.join(label_dirname, dire, 'Camera 5')):
#         labels.append(os.path.join(label_dirname, dire, 'Camera 5/', dir))
#     for dir in os.listdir(os.path.join(label_dirname, dire, 'Camera 6')):
#         labels.append(os.path.join(label_dirname, dire, 'Camera 6/', dir))
# label_dirname = os.path.join(file_path, 'Label_road03/Label/')
# for dire in os.listdir(label_dirname):
#     for dir in os.listdir(os.path.join(label_dirname, dire, 'Camera 5')):
#         labels.append(os.path.join(label_dirname, dire, 'Camera 5/', dir))
#     for dir in os.listdir(os.path.join(label_dirname, dire, 'Camera 6')):
#         labels.append(os.path.join(label_dirname, dire, 'Camera 6/', dir))
# label_dirname = os.path.join(file_path, 'Label_road04/Label/')
# for dire in os.listdir(label_dirname):
#     for dir in os.listdir(os.path.join(label_dirname, dire, 'Camera 5')):
#         labels.append(os.path.join(label_dirname, dire, 'Camera 5/', dir))
#     for dir in os.listdir(os.path.join(label_dirname, dire, 'Camera 6')):
#         labels.append(os.path.join(label_dirname, dire, 'Camera 6/', dir))
#
# images = list()
# for ln in labels:
#     if 'Label_road02' in ln:
#         img_name = ln.replace('Label_road02/Label/', 'ColorImage_road02/ColorImage/')
#         img_name = img_name.replace('_bin.png', '.jpg')
#     elif 'Label_road03' in ln:
#         img_name = ln.replace('Label_road03/Label/', 'ColorImage_road03/ColorImage/')
#         img_name = img_name.replace('_bin.png', '.jpg')
#     elif 'Label_road04' in ln:
#         img_name = ln.replace('Label_road04/Label/', 'ColorImage_road04/ColorImage/')
#         img_name = img_name.replace('_bin.png', '.jpg')
#     images.append(img_name)
#
# print(len(labels))
# print(len(images))
# c ={'image':images,'label':labels}#合并成一个新的字典c
# data = pd.DataFrame(c)
# print(data.head())
# data.to_csv('./train.csv')
# --------------------创建训练样本/验证样本/测试样本----------------------------------------
# data=pd.read_csv('./train.csv')
# images=data['image']
# labels=data['label']
#
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size=0.4)
# X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=0.25)
#
#
# c ={'image':X_train,'label':y_train}#合并成一个新的字典c
# datax = pd.DataFrame(c)
# datax.to_csv('./train.csv')
#
# c ={'image':X_val,'label':y_val}#合并成一个新的字典c
# datay = pd.DataFrame(c)
# datay.to_csv('./val.csv')
#
# c ={'image':X_test,'label':y_test}#合并成一个新的字典c
# dataz = pd.DataFrame(c)
# dataz.to_csv('./test.csv')

data=pd.read_csv('./train.csv')
images=data['image']
print(len(images))

data=pd.read_csv('./val.csv')
images=data['image']
print(len(images))

data=pd.read_csv('./test.csv')
images=data['image']
print(len(images))
