import torch
import numpy as np
import time
import os
import argparse
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

from models.segnet import SegNet
from models.resnet_unet import ResNet34Unet
from models.loss import DiscriminativeLoss
from Baidureader import BaiduDataset
from utils.logger import Logger
from utils.util import label_accuracy_score
from tqdm import tqdm
import torchvision.transforms as transforms

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 9  #label类
LEARNING_RATE = 1e-3 #1e-5
BATCH_SIZE = 2
NUM_EPOCHS = 50 #100
LOG_INTERVAL = 50
# SIZE = [224, 224]
# SIZE = [1280,704]  #W×H
SIZE = [1024, 384]
def train():
    # refer from : https://github.com/Sayan98/pytorch-segnet/blob/master/src/train.py
    is_better = True
    prev_loss = float('inf')
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        t_start = time.time()
        loss_f = []

        for batch_idx, (imgs, sem_labels, ins_labels) in tqdm(enumerate(train_dataloader)):
            loss = 0
            img_tensor = torch.autograd.Variable(imgs).cuda()
            sem_tensor = torch.autograd.Variable(sem_labels).cuda()
            ins_tensor = torch.autograd.Variable(ins_labels).cuda()

            # Init gradients
            optimizer.zero_grad()

            # Predictions
            sem_pred, ins_pred = model(img_tensor)

            # Discriminative Loss
            disc_loss = criterion_disc(ins_pred, ins_tensor, [5] * len(img_tensor))
            loss += disc_loss

            # CrossEntropy Loss

            ce_loss = criterion_ce(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS),
                                   sem_tensor.view(-1))
            loss += ce_loss

            loss.backward()
            optimizer.step()

            loss_f.append(loss.cpu().data.numpy())

            if batch_idx % LOG_INTERVAL == 0:
                print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

                #Tensorboard
                info = {'loss': loss.item(), 'ce_loss': ce_loss.item(), 'disc_loss': disc_loss.item(), 'epoch': epoch}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, batch_idx + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx + 1)
                    # logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx + 1)

                # 3. Log training images (image summary)
                sem_show=sem_pred.max(1)[1]
                info = {'images': img_tensor.view(-1, 3, SIZE[1], SIZE[0])[:BATCH_SIZE].cpu().numpy(),
                        'labels': sem_tensor.view(-1, SIZE[1], SIZE[0])[:BATCH_SIZE].cpu().numpy(),
                        # 'sem_preds': sem_pred.view(-1, 2, SIZE[1], SIZE[0])[:BATCH_SIZE,1].data.cpu().numpy(),
                        'sem_preds': sem_show.view(-1, SIZE[1], SIZE[0])[:BATCH_SIZE].data.cpu().numpy(),
                        'ins_preds': ins_pred.view(-1, SIZE[1], SIZE[0])[:BATCH_SIZE*5].data.cpu().numpy()}

                for tag, images in info.items():
                    logger.image_summary(tag, images, batch_idx + 1)

        dt = time.time() - t_start
        is_better = np.mean(loss_f) < prev_loss
        # scheduler.step()

        if is_better:
            prev_loss = np.mean(loss_f)
            print("\t\tBest Model.")
            torch.save(model.state_dict(), "model_best.pth")
            torch.save(optimizer.state_dict(),'optimizer_best.pth')

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s, Lr: {:2f}".format(epoch+1, np.mean(loss_f), dt, optimizer.param_groups[0]['lr']))
        # cv(epoch)



def predict(im): # 预测结果
    im = im.unsqueeze(0)
    out, ins_pred = model(im)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    return pred

def cv(batch_idx):
    model.eval()
    test_num=len(val_dataset)
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    for step, (imgs, sem_labels, ins_labels) in (enumerate(val_dataloader)):
    # for step in tqdm(range(500)):
    #     (imgs, sem_labels, ins_labels)=next(iter(train_dataloader))
        test_data, test_label, _ = imgs[0], sem_labels[0], ins_labels
        pred = predict(test_data.cuda())
        label_true = test_label.data.cpu().numpy()
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label_true, pred, 9)
        # break
        # for lbt, lbp in zip(label_true, pred):
        #     acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 9)
        eval_acc += acc
        eval_acc_cls += acc_cls
        eval_mean_iu += mean_iu
        eval_fwavacc += fwavacc
        if step % np.floor(test_num*0.2)==0 : print("进度:",step/test_num);
        if step==test_num:
            # Tensorboard
            info = {'acc': eval_acc/test_num, 'acc_cls': eval_acc_cls/test_num, 'mean_iu': eval_mean_iu/test_num, 'fwavacc': eval_fwavacc/test_num}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, batch_idx + 1)
            print(eval_acc/test_num,eval_acc_cls/test_num,eval_mean_iu/test_num,eval_fwavacc/test_num)
            break
    model.train()


if __name__ == "__main__":
   logger = Logger('./logs')
   train_path = './data/train.csv'
   val_path='./data/val.csv'
   transformations = transforms.Compose([
       # transforms.Resize(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])  # 将读取的图片变化
   # train_path = '/home/cxd/python/pycharm/data/train_set/'
   train_dataset = BaiduDataset(train_path,transformations=transformations, size=SIZE)
   train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

   val_dataset = BaiduDataset(val_path,transformations=transformations , size=SIZE)
   val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
   # model = SegNet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda()
   # model = ENet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda()
   # model = Unet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda()
   model = ResNet34Unet(OUTPUT_CHANNELS).cuda()
   # model.init()
   if os.path.isfile("model_best.pth"):
       print("Loaded model_best.pth")
       model.load_state_dict(torch.load("model_best.pth"))

   pinlv=np.load('./data/Baidu_freq.npy')
   weight=(1/np.log(pinlv+1.02))
   weight=torch.from_numpy(weight).cuda().float()
   criterion_ce = torch.nn.CrossEntropyLoss(weight=weight).cuda()
   criterion_disc = DiscriminativeLoss(delta_var=0.1,
                                       delta_dist=0.6,
                                       norm=2,
                                       usegpu=True).cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   if os.path.isfile("optimizer_best.pth"):
       print("Loaded optimizer_best.pth")
       optimizer.load_state_dict(torch.load("optimizer_best.pth"))
   # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3,4,5,6,7], gamma=0.5)

   train()
