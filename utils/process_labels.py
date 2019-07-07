"""
Author: Jingxiao Gu
Baidu Account: Seigato
Description: Encode & Decode Labels for Lane Segmentation Competition
NOTE: For 8 classes
"""
import numpy as np
from collections import namedtuple
Label = namedtuple( 'Label' , ['name' ,'id','trainId','category','categoryId','hasInstances','ignoreInEval','color'] )
Blabels = [
# name id trainId category catId hasInstances ignoreInEval color
Label( 'void' , 0 , 0, 'void' , 0 , False , False , ( 0, 0, 0) ),
Label( 's_w_d' , 200 , 1 , 'dividing' , 1 , False , False , ( 70, 130, 180) ),
Label( 's_y_d' , 204 , 1 , 'dividing' , 1 , False , False , (220, 20, 60) ),
# Label( 'ds_w_dn' , 213 , 1 , 'dividing' , 1 , False , True , (128, 0, 128) ),
Label( 'ds_y_dn' , 209 , 1 , 'dividing' , 1 , False , False , (255, 0, 0) ),
# Label( 'sb_w_do' , 206 , 1 , 'dividing' , 1 , False , True , ( 0, 0, 60) ),
# Label( 'sb_y_do' , 207 , 1 , 'dividing' , 1 , False , True , ( 0, 60, 100) ),
Label( 'b_w_g' , 201 , 2 , 'guiding' , 2 , False , False , ( 0, 0, 142) ),
Label( 'b_y_g' , 203 , 2 , 'guiding' , 2 , False , False , (119, 11, 32) ),
# Label( 'db_w_g' , 211 , 2 , 'guiding' , 2 , False , True , (244, 35, 232) ),
# Label( 'db_y_g' , 208 , 2 , 'guiding' , 2 , False , True , ( 0, 0, 160) ),
# Label( 'db_w_s' , 216 , 3 , 'stopping' , 3 , False , True , (153, 153, 153) ),
Label( 's_w_s' , 217 , 3 , 'stopping' , 3 , False , False , (220, 220, 0) ),
# Label( 'ds_w_s' , 215 , 3 , 'stopping' , 3 , False , True , (250, 170, 30) ),
# Label( 's_w_c' , 218 , 4 , 'chevron' , 4 , False , True , (102, 102, 156) ),
# Label( 's_y_c' , 219 , 4 , 'chevron' , 4 , False , True , (128, 0, 0) ),
Label( 's_w_p' , 210 , 5 , 'parking' , 5 , False , False , (128, 64, 128) ),
# Label( 's_n_p' , 232 , 5 , 'parking' , 5 , False , True , (238, 232, 170) ),
Label( 'c_wy_z' , 214 , 6 , 'zebra' , 6 , False , False , (190, 153, 153) ),
# Label( 'a_w_u' , 202 , 7 , 'thru/turn' , 7 , False , True , ( 0, 0, 230) ),
Label( 'a_w_t' , 220 , 7 , 'thru/turn' , 7 , False , False , (128, 128, 0) ),
Label( 'a_w_tl' , 221 , 7 , 'thru/turn' , 7 , False , False , (128, 78, 160) ),
Label( 'a_w_tr' , 222 , 7 , 'thru/turn' , 7 , False , False , (150, 100, 100) ),
# Label( 'a_w_tlr' , 231 , 7 , 'thru/turn' , 7 , False , True , (255, 165, 0) ),
Label( 'a_w_l' , 224 , 7 , 'thru/turn' , 7 , False , False , (180, 165, 180) ),
Label( 'a_w_r' , 225 , 7 , 'thru/turn' , 7 , False , False , (107, 142, 35) ),
Label( 'a_w_lr' , 226 , 7 , 'thru/turn' , 7 , False , False , (201, 255, 229) ),
# Label( 'a_n_lu' , 230 , 7 , 'thru/turn' , 7 , False , True , (0, 191, 255) ),
# Label( 'a_w_tu' , 228 , 7 , 'thru/turn' , 7 , False , True , ( 51, 255, 51) ),
# Label( 'a_w_m' , 229 , 7 , 'thru/turn' , 7 , False , True , (250, 128, 114) ),
# Label( 'a_y_t' , 233 , 7 , 'thru/turn' , 7 , False , True , (127, 255, 0) ),
Label( 'b_n_sr' , 205 , 8 , 'reduction' , 8 , False , False , (255, 128, 0) ),
# Label( 'd_wy_za' , 212 , 8 , 'attention' , 8 , False , True , ( 0, 255, 255) ),
Label( 'r_wy_np' , 227 , 8 , 'no parking' , 8 , False , False , (178, 132, 190) ),
# Label( 'vom_wy_n' , 223 , 8 , 'others' , 8 , False , True , (128, 128, 64) ),
Label( 'om_n_n' , 250 , 8 , 'others' , 8 , False , False , (102, 0, 204) ),
# Label( 'noise' , 249 , 0 , 'ignored' , 0 , False , True , ( 0, 153, 153) ),
# Label( 'ignored' , 255 , 0 , 'ignored' , 0 , False , True , (255, 255, 255) ),
]

encode = np.zeros(256)  # 每个像素点有 0 ~ 255 的选择
decode = np.zeros(9)  # label: 0 ~ 8

def code_init():
    for obj in Blabels:
        encode[obj.id] = obj.trainId
    for obj in Blabels:
        decode[obj.trainId] = obj.id

def encode_labels(color_mask):
    #color_mask标签--->one-hot编码

    encode_mask = encode[color_mask]
    return encode_mask

def decode_labels(labels):
    # one-hot编码--->color_mask标签
    deocde_mask = decode[labels]

    return deocde_mask

def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask

def verify_labels(labels):
    pixels = [0]
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            pixel = labels[x, y]
            if pixel not in pixels:
                pixels.append(pixel)
    print('The Labels Has Value:', pixels)


# from scipy import ndimage as ndi
# from sklearn.cluster import DBSCAN
# def coloring(mask):
#     ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#     n_ins = len(np.unique(mask)) - 1
#     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
#     for i in range(n_ins):
#         ins_color_img[mask == i + 1] =\
#             (np.array(colors[i][:3]) * 255).astype(np.uint8)
#     return ins_color_img
#
#
#
# def gen_instance_mask(sem_pred, ins_pred, n_obj):
#     embeddings = ins_pred[:, sem_pred>=1].transpose(1, 0)
# #     clustering = KMeans(n_obj).fit(embeddings)
#     clustering = DBSCAN(eps=0.05).fit(embeddings)
#     labels = clustering.labels_
#
#     print(sem_pred.shape,ins_pred.shape)
#     print(embeddings.shape)
#     print(labels.shape)
#     instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
#     for i in range(n_obj):
#         lbl = np.zeros_like(labels, dtype=np.uint8)
#         lbl[labels == i] = i + 1
#         instance_mask[sem_pred] += lbl
#
#     return instance_mask
