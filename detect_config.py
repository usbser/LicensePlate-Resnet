from models.detect_net import WpodNet
import torch
image_root = '/home/cq/public/hibiki/CCPD2019/ccpd_base'
batch_size = 96
weight = 'weights/wpod_net.pt'
epoch = 100000
net = WpodNet
device = 'cuda:1'
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else 'cpu')



