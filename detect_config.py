from models.detect_net import WpodNet
import torch
image_root = 'D:\\dataSet\\CCPD2019\\5000_images\\train'
#image_root = '/data/jy/CCPD2019/train'
batch_size = 16
weight = 'weights/wpod_net.pt'
weight_explore = 'weights/wpod_net-300epoch-prune-100train-5000image-50fineturn.pt'
weight_explore_prune ='weights/wpod_net-300epoch-prune-100train-5000image-50fineturn.pt'
epoch = 200
net = WpodNet
device = 'cuda:0'
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else 'cpu')



