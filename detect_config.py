from models.detect_net import WpodNet
import torch
#image_root = 'D:\\dataSet\\CCPD2019\\5000_images\\test'
image_root = '/data/jy/CCPD2019/train'
batch_size = 256
weight = 'weights/wpod_net.pt'
epoch = 300
net = WpodNet
device = 'cuda:4'
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else 'cpu')



