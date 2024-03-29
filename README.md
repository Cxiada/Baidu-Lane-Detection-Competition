百度常规挑战赛-车道线检测解决方案

无人车车道线检测挑战赛](http://aistudio.baidu.com/aistudio/#/competition/detail/5)，最终的平均交并比为0.43（排名17/743），ID：_表妹酱。

比赛内容为：根据百度提供的车道线数据和车道线标注图像，完成车道线检测，属于语义分割问题，加上背景类共有9个类别。

### 数据集处理

我的设备是一块1080Ti，可用显存11G。

【1】本次比赛图像分辨率非常的大，3384x1710，首先裁掉了图片最上部的图像，因为摄像头的拍摄角度原因，这部分几乎不存在车道线。

【2】数据预处理一开始主要包括图像直方图均衡化，随机翻转，随机对比度增强，加噪声。但是在后面进行inference时发现训练集road3和road4中好多图片要么太暗要太亮，不符合测试集的分布，于是决定对每张图片都进行亮度增强，然后把错误的图片从训练集中删除。

【3】将图像resize到1024x512的分辨率进行训练，不使用随机裁剪的方法，是由于感受野太小，而车道线很长，容易出现车道线被截断的情况，对特征的提取很不利。

**TO DO：** 看了大佬的想法，<u>使用了多分辨率（768x256,1024x384,1536x512）进行训练，并且大分辨率的模型是基于前一个小分辨率的预训练模型而来</u>。用低分辨率模型进行预训练的想法非常棒，即可以快速验证网络，又可以加速高分辨率网络的训练。

【4】数据类不平衡问题，由于各个类的占比很悬殊，使用了加权交叉熵作为Loss，权值为

$$
\frac{1}{ln(p_{class}+c)}
$$

**TO DO：**看了大佬的想法，<u>由于第五类和第八类训练数据较少（但是测试集中占比不少），所以针对这两类，采用了iaa的图像处理，从亮度、饱和度、噪点、对比度、crop、scale等方面做了共计12000张图片的增强。</u>也可以单独对难预测的类别进行训练。

### 训练网络

【5】本次比赛使用了resnet残差网络+unet网络架构。基本上比赛高分都是使用unet网络，这个网络训练简单，能同时提取深层特征和浅层特征。

**TO DO：**由于在实验室主要处理的是雷达图像、遥感图像。希望在比赛中学习摄像头光学图像处理的经验，没有去训练多个网络。根据kaggle的比赛top选手分享的经验来看，要提高成绩一般都采用多个训练网络来求平均结果。

【6】由于比赛得分是计算平均交并比，其中某些在训练集中出现频率低，本身在图片中面积非常小的类在预测时的交并比非常差，会导致整体得分被严重拉低。在原有的加权交叉熵的基础上，增加另一个聚类Loss（Discriminative Loss）。

$$
类内方差：L_{var}=\frac{1}{C}\sum\frac{1}{N_c}\sum[\lvert\lvert u_c-x_i \lvert\lvert-\delta_v]_+^2
$$

$$
类中心距离：L_{dist}=\frac{1}{C(C-1)}\sum\sum[\delta_d-\lvert\lvert u_{ca}-u_{cb} \lvert\lvert]_+^2
$$

$$
正则项：L_{reg}=\frac{1}{C}\sum\lvert\lvert u_c\lvert\lvert
$$


$$
聚类Loss=\alpha L_{reg}+\beta L_{dist}+\gamma L_{reg}
$$


聚类损失函数相当于将车道线映射到嵌入空间，在聚类损失函数的作用下，能改善小类的预测情况。由于聚类是不带标签的，最后聚类得到的类属要自己贴标签。可以根据交并比测试结果决定类别的归属。

将网络改为：

```
                            
                           |------>特征解码（B×H×W×C）使用加权交叉熵
                           |
  输入图片----->共享特征提取--|
                           |
                           |------>嵌入空间（B×H×W×S）使用聚类Loss
```

【7】在训练策略上，先采用了Cycle LR策略，采用Adam，加速模型的收敛，作为预训练参数。然后在经过几个epoch后，开始手动调节学习率。

【8】后处理操作：因为采用的不是随机裁剪来训练网络，需要将测试集图片resize到训练时的分辨率进行inference。再将预测的图片resize回到原图片尺寸，这里使用的是bilinear进行缩放。这里使用全连接条件随机场（dense crfs）来对缩放后的图片进行画质优化，降低插值算法带来的预测误差。

### 训练模型结果

### 环境说明

```
Python 3.6
pytorch   0.4.1
opencv-python 3.4.2
scikit-learn  0.19.2
pydensecrf  2.2

```

