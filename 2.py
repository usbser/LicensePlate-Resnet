# Copyright (c) 2019, RangerUFO
#
# This file is part of alpr_utils.
#
# alpr_utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpr_utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alpr_utils.  If not, see <https://www.gnu.org/licenses/>.

import time
# import mxnet as mx
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data
import cv2
from read_plate import ReadPlate
from PIL import Image, ImageDraw, ImageFont
import numpy
from utils.drawchinese import DrawChinese


# def load_image(path):
#     with open(path, "rb") as f:
#         buf = f.read()
#     return mx.image.imdecode(buf)


def fixed_crop(raw, bbox):
    x0 = max(int(bbox[0].asscalar()), 0)
    x0 = min(int(x0), raw.shape[1])
    y0 = max(int(bbox[1].asscalar()), 0)
    y0 = min(int(y0), raw.shape[0])
    x1 = max(int(bbox[2].asscalar()), 0)
    x1 = min(int(x1), raw.shape[1])
    y1 = max(int(bbox[3].asscalar()), 0)
    y1 = min(int(y1), raw.shape[0])
    return raw[x0:x1,y0:y1]
    #return mx.image.fixed_crop(raw, x0, y0, x1 - x0, y1 - y0)


def test(images):
    #context = mx.cpu(0)
    yes = 0
    count = 0
    yesss = 0
    yolo = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)#, ctx=context
    read_plate = ReadPlate()
    for path in images:

        label = path.split('/')[-1].split('_')[0]
        # print(label)
        # exit()
        '''加载图片'''
        # raw = load_image(path)
        raw = cv2.imread(path)
        # print(raw.shape)
        ts = time.time()
        # print('aaaaaaaaaaaaa')
        '''图片归一化'''
        x, _ = data.transforms.presets.yolo.transform_test(raw, short=512)
        # print(x)
        '''得到侦测结果'''
        classes, scores, bboxes = yolo(x)
        # print(classes.shape)
        '''反算回归框'''
        bboxes[0, :, 0::2] = bboxes[0, :, 0::2] / x.shape[3] * raw.shape[1]
        bboxes[0, :, 1::2] = bboxes[0, :, 1::2] / x.shape[2] * raw.shape[0]
        vehicles = [
            fixed_crop(raw, bboxes[0, i]) for i in range(classes.shape[1])
            if (yolo.classes[int(classes[0, i].asscalar())] == 'car' or
                yolo.classes[int(classes[0, i].asscalar())] == 'bus') and
               scores[0, i].asscalar() > 0.5
        ]
        # print(vehicles)
        # exit()
        # print("yolo profiling: %f" % (time.time() - ts))
        for i, raw in enumerate(vehicles):
            # print("vehicle[%d]:" % i)
            # print(raw)
            image = raw.asnumpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('a', image)
            # cv2.waitKey()
            count += 1
            '''侦测网络、字符变量、字符识别网络、图片、样本尺寸、阈值、车牌高宽（48，144）、使用定向搜索，定向尺寸、设备'''
            results = read_plate(image)
            for plate, box in results:
                print(yes,yesss,count,yes/count,yesss/count,label,plate)
                if label == plate:
                    yes+=1
                if label[1:]==plate[1:]:
                    yesss+=1
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                image = DrawChinese(image, plate, (int(box[0]), int(box[1])-50), 40,(200,0,0))
                cv2.imshow('a',image)
                cv2.waitKey(0)
                break
            break
    print(yes,count,yes/count)


if __name__ == "__main__":
    import os

    images = []
    # for file_name in os.listdir('/home/cq/public/hibiki/CCPD2019/test'):
    #     # for image_name in os.listdir(f'/home/cq/public/hibiki/CCPD2019/ccpd_db/{file_name}'):
    #     images.append(f'/home/cq/public/hibiki/CCPD2019/test/{file_name}')

    for file_name in os.listdir('D:\\dataSet\\CCPD2019\\5000_images\\test'):
        # for image_name in os.listdir(f'/home/cq/public/hibiki/CCPD2019/ccpd_db/{file_name}'):
        images.append(f'D:\\dataSet\\CCPD2019\\5000_images\\test\\{file_name}')
    test(images)
