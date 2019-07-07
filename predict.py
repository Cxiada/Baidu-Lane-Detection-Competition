import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import os
import torch as t
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
# from mxtorch import transforms as tfs
import  torchvision.transforms.functional as tf
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.utils import make_grid,save_image
from viz import make_dot,Visualizer
# from graphviz import Digraph
from utils import label_accuracy_score
from dataset.dataset import DatasetFromjpg,CLASSES,COLORMAP,DatasetFrombaidu
from dataset.CamVidDataset import CamVidDataset,CamVid_colours
from models import FcnResNet,Fcnvgg16,FCN8s,SegNet
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  # 将读取的图片变化

# cv_dataset = DatasetFromjpg('./VOC2012/', mold='val', transforms=transformations, output_size=(320,320),predct=True)
# test_loader = DataLoader(dataset=cv_dataset, batch_size=10, shuffle=True, num_workers=2)
# cv_dataset = CamVidDataset('./CamVid/', mold='test', transforms=transformations, output_size=(352, 480), predct=True)
cv_dataset = DatasetFrombaidu('./baidu/', mold='val', transforms=transformations, output_size=(720, 720),predct=True)
# cv_loader = DataLoader(dataset=cv_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)
# 定义预测函数
# print(cv_dataset.datalen)
# cm = np.array(COLORMAP).astype('uint8')
# cm = np.array(CamVid_colours).astype('uint8')
n_class=9
net=SegNet(num_classes=9)
# net=SegNet(num_classes=12)
net.cuda()
net.eval()
dir='./checkpoints/baiduSegNet5.pth'
state=t.load(dir)
net.load_state_dict(state['net'])
test_data, test_label = cv_dataset[1]
print(test_data.size())


# out=net(Variable(test_data.unsqueeze(0)).cuda())
# print(out.data.size())
# pred = out.max(1)[1].squeeze().cpu().data.numpy()
# print(pred.shape)


def predict(im, label):  # 预测结果
    im = Variable(im.unsqueeze(0)).cuda()
    out = net(im)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    # pred = cm[pred]

    return pred, label
# test_data, test_label = cv_dataset[81]
# pred, label=predict(test_data, test_label)
# plt.subplot(3,1,1)
# plt.imshow(Image.open(cv_dataset.datapath[81]))
# plt.subplot(3,1,2)
# plt.imshow(label)
# plt.subplot(3,1,3)
# plt.imshow(pred)
# plt.show()
# _, figs = plt.subplots(4, 2, figsize=(12, 10))
a, label=predict(test_data[:,:,0:839], test_label[:,0:839])
print(a.shape)
pred, label=predict(test_data[:,:,840:1679], test_label[:,0:839])
print(pred.shape)
a=np.concatenate((a,pred), axis=1)
print(a.shape)
pred, label=predict(test_data[:,:,1680:2519], test_label[:,0:839])
a=np.concatenate((a,pred), axis=1)
print(a.shape)
pred, label=predict(test_data[:,:,2520:3359], test_label[:,0:839])
a=np.concatenate((a,pred), axis=1)
a=a.astype('uint8')
img=Image.fromarray(a)
img = img.resize((3384, 1710))
b=np.array(img,'int64')
# acc, acc_cls, mean_iu, fwavacc=label_accuracy_score(b, test_label.squeeze(), n_class)
# print(mean_iu)
print(b.shape, test_label.shape)
plt.imshow(img)
plt.show()
# img.save('test.png')
# for i in range(4):
#     test_data, test_label = cv_dataset[1]
#     # print(test_data.size())
#     # print(test_label.shape)
#     pred, label=predict(test_data[:,:,0:839], test_label[:,0:839])

#     figs[i, 0].imshow(Image.open(cv_dataset.datapath[i]))
#     figs[i, 0].axes.get_xaxis().set_visible(False)
#     figs[i, 0].axes.get_yaxis().set_visible(False)
#     figs[i, 1].imshow(label)
#     figs[i, 1].axes.get_xaxis().set_visible(False)
#     figs[i, 1].axes.get_yaxis().set_visible(False)
#     print(pred.shape,'   ',  label.shape  )
#     figs[i, 2].imshow(pred)
#     figs[i, 2].axes.get_xaxis().set_visible(False)
#     figs[i, 2].axes.get_yaxis().set_visible(False)
# plt.show()
# _, figs = plt.subplots(6, 3, figsize=(12, 10))
# for i in range(6):
#     test_data, test_label = cv_dataset[i]
#     pred, label=predict(test_data, test_label)
#     figs[i, 0].imshow(Image.open(cv_dataset.datapath[i]))
#     figs[i, 0].axes.get_xaxis().set_visible(False)
#     figs[i, 0].axes.get_yaxis().set_visible(False)
#     figs[i, 1].imshow(label)
#     figs[i, 1].axes.get_xaxis().set_visible(False)
#     figs[i, 1].axes.get_yaxis().set_visible(False)
#     print(pred.shape,'   ',  label.shape  )
#     figs[i, 2].imshow(pred)
#     figs[i, 2].axes.get_xaxis().set_visible(False)
#     figs[i, 2].axes.get_yaxis().set_visible(False)
# plt.show()