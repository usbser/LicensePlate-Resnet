import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import argparse
import time

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.distributed as dist
from utils.loss import FocalLossManyClassification
from utils.dataset import DetectDataset
from einops import rearrange
import detect_config as config
import torch

# # 1. 初始化group，参数backend默认为nccl，表示多gpu，多cpu是其它参数
# dist.init_process_group(backend='nccl')
# # 2. 添加一个local_rank参数
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank")
# args = parser.parse_args()
# # 3. 从外面得到local_rank参数，在调用DDP的时候，其会根据调用gpu自动给出这个参数
# local_rank = args.local_rank
# # 4.根据local_rank指定使用那块gpu
# torch.cuda.set_device(local_rank)
# DEVICE = torch.device("cuda", local_rank)

class Trainer:

    def __init__(self):
        self.net = config.net()
        self.epoch = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00001)
        if os.path.exists(config.weight):
            ckpt = torch.load(config.weight, map_location='cpu')
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if self.net.state_dict()[k].shape == v.shape}
            self.net.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epoch = ckpt['epoch']
            print('成功加载网络参数')
        else:
            print('未加载网络参数')

        self.l1_loss = nn.L1Loss()
        self.c_loss = nn.CrossEntropyLoss()
        self.dataset = DetectDataset()
        # sampler = torch.utils.data.DistributedSampler(self.dataset, num_replicas=2,
        #                                               rank=dist.get_rank(), shuffle=True,
        #                                               drop_last=True)
        # self.data_loader = DataLoader(self.dataset, batch_size=8, num_workers=4, pin_memory=True,
        #                     sampler=sampler, shuffle=False, collate_fn=None)
        self.data_loader = DataLoader(self.dataset, config.batch_size, drop_last=True)
        self.net.to(config.device)
        # self.net = self.net.to(local_rank)
        # self.net = ddp(self.net , device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)


    def train(self):
        s = time.time()
        for epoch in range(self.epoch, config.epoch):
            self.net.train()
            loss_sum = 0
            for i, (images, labels) in enumerate(self.data_loader):
                images = images.to(config.device)
                labels = labels.to(config.device)

                predict = self.net(images)
                loss_c, loss_p = self.count_loss(predict, labels)
                loss = loss_c + loss_p
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i%10==0:
                    print('epoch',epoch,'batch',i,'loss:',loss.item(),'loss_c:',loss_c.item(),'loss_p:',loss_p.item())
                loss_sum += loss.item()
            logs = f'epoch:{epoch}, loss:{loss_sum / len(self.data_loader)} ,time:{round(time.time()-s)} secs'
            print(logs)
            final_epoch = epoch + 1 == config.epoch
            #save model
            ckpt = {'epoch':epoch,
                    'model':self.net,
                    'optimizer':None if final_epoch else self.optimizer.state_dict()}
            #torch.save(self.net.state_dict(), config.weight)
            torch.save(ckpt, config.weight)
            tb_writer.add_scalar("loss",epoch,loss_sum)
            tb_writer.add_scalar("time", epoch, time.time()-s)

    def count_loss(self, predict, target):
        condition_positive = target[:, :, :, 0] == 1
        condition_negative = target[:, :, :, 0] == 0

        predict_positive = predict[condition_positive]
        predict_negative = predict[condition_negative]

        target_positive = target[condition_positive]
        target_negative = target[condition_negative]
        # print(target_positive.shape)
        n, v = predict_positive.shape
        if n > 0:   #predict  8个数  前两个代表真样本和负样本的概率，，后六个为仿射系数 用于将所在区域转化为目标框   计算所有正样本的交叉熵
            loss_c_positive = self.c_loss(predict_positive[:, 0:2], target_positive[:, 0].long())
        else:
            loss_c_positive = 0
        loss_c_nagative = self.c_loss(predict_negative[:, 0:2], target_negative[:, 0].long())#计算所有负样本的交叉熵
        loss_c = loss_c_nagative + loss_c_positive  #总的预测的交叉熵

        if n > 0:
            affine = torch.cat( #删掉预测结果的头两个参数
                (
                    predict_positive[:, 2:3],
                    predict_positive[:,3:4],
                    predict_positive[:,4:5],
                    predict_positive[:,5:6],
                    predict_positive[:,6:7],
                    predict_positive[:,7:8]
                ),
                dim=1
            )
            # print(affine.shape)
            # exit()
            trans_m = affine.reshape(-1, 2, 3) #把后面六个值变成2*3矩阵
            unit = torch.tensor([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]).transpose(0, 1).to(
                trans_m.device).float()  #代表(-w/2,-h/2)、(w/2,-h/2)、(w/2,h/2)、(-w/2,h/2)
            # unit  用于计算预测的边界框和实际边界框之间的IoU  [n,2,3]*[3,4]   = [n,2,4],预测的6个值转换为4个点坐标
            point_pred = torch.einsum('n j k, k d -> n j d', trans_m, unit)
            point_pred = rearrange(point_pred, 'n j k -> n (j k)')
            loss_p = self.l1_loss(point_pred, target_positive[:, 1:])
        else:
            loss_p = 0
        # exit()
        return loss_c, loss_p

        # return loss


if __name__ == '__main__':
    trainer = Trainer()
    tb_writer = SummaryWriter(comment='train_results')
    trainer.train()
    tb_writer.close()
