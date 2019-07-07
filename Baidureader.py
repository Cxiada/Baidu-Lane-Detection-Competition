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
import torch
import pandas as pd
from utils.process_labels import code_init,encode_labels,decode_labels,decode_color_labels,verify_labels
from utils.image_process import crop_resize_data,CLAHE_nomalization,contrast,random_filp


class BaiduDataset(data.Dataset):
    # refer from :
    # https://github.com/vxy10/ImageAugmentation
    # https://github.com/TuSimple/tusimple-benchmark/blob/master/example/lane_demo.ipynb
    def __init__(self, file_path, transformations, size=[640, 360], train=True):
        warnings.simplefilter("ignore")
        code_init()

        data=pd.read_csv(file_path)
        m_labels=data["label"]
        m_images = data["image"]
        self.transforms = transformations
        self.labels = list()
        for label in m_labels:
            self.labels.append(label)

        self.images = list()
        for image in m_images:
            self.images.append(image)

        self.n_seg = 9
        self.file_path = file_path
        self.flags = {'size': size, 'train': train}

        self.len = len(self.labels)

    def get_lane_image(self, idx):
        #  self.img
        #  self.label_img 语义分割标签
        #  self.ins_img   嵌入空间，每个类一个平面
        ln=self.labels[idx]
        img_name = self.images[idx]

        self.img=cv2.imread(img_name,1)    #uint8
        self.label_img = cv2.imread(ln,0)  #uint8
        # self.label_img = np.array(self.label_img, dtype=np.int32)
        self.label_img = encode_labels(self.label_img)

    def get_ins_img(self):
        height, width = self.label_img.shape
        self.ins_img = np.zeros((0, height, width), np.uint8)
        for i in range(self.n_seg):
            gt = np.array(self.label_img==i, dtype=np.uint8)
            self.ins_img = np.concatenate([self.ins_img, gt[np.newaxis]])


    def __getitem__(self, idx):
        self.get_lane_image(idx)
        self.img, self.label_img=crop_resize_data(self.img, self.label_img)
        self.img=CLAHE_nomalization(self.img)
        self.img = contrast(self.img)
        self.img, self.label_img=random_filp(self.img, self.label_img)
        self.label_img = self.label_img * (self.label_img < 9) * (self.label_img>=0)
        self.get_ins_img()

        # cv2.imshow('img', self.img)
        # cv2.imshow('label', self.label_img)
        # cv2.imshow('ins', self.ins_img[2]*100)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.flags['train']:
            # self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            self.img=self.transforms(self.img)
            self.label_img = np.array(self.label_img, dtype=np.float32)
            self.ins_img = np.array(self.ins_img, dtype=np.float32)
            return torch.Tensor(self.img), torch.LongTensor(self.label_img), torch.Tensor(self.ins_img)
        else:
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            return torch.Tensor(self.img)

    def __len__(self):
        return self.len


