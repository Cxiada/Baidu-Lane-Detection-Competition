import cv2
import numpy as np
# import os
# import six
# import random
# from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
# import torch
import pandas as pd
from utils.process_labels import code_init,encode_labels,decode_labels,decode_color_labels,verify_labels
from utils.image_process import crop_resize_data,CLAHE_nomalization,contrast,random_filp
from Baidureader import BaiduDataset
data=BaiduDataset('./train.csv', [1024, 384])
image,label,ins=data[0]
print(image.size())
print(label.size())
label_=np.array(label,dtype='uint8')
verify_labels(label_)
decode_mask=decode_color_labels(label_)
plt.imshow(decode_mask.transpose(1,2,0))
plt.show()

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