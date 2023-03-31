import os
import time

from detect_explorer import DExplorer
from ocr_explorer import Explorer
import ocr_config as config
import cv2
import numpy
from utils.drawchinese import DrawChinese

class ReadPlate:
    """
    读取车牌号
    传入侦测到的车辆图片，即可识别车牌号。
    返回：
        [[车牌号，回归框],……]
    """
    def __init__(self):
        self.detect_exp = DExplorer()
        self.ocr_exp = Explorer()

    def __call__(self, image):
        points = self.detect_exp(image)
        h, w, _ = image.shape
        result = []
        # print(points)
        for point, _ in points:
            plate, box = self.cutout_plate(image, point)
            # print(box)
            lp = self.ocr_exp(plate)
            result.append([lp, box])
            # cv2.imshow('a', plate)
            # cv2.waitKey()
        return result

    def cutout_plate(self, image, point):
        h, w, _ = image.shape
        x1, x2, x3, x4, y1, y2, y3, y4 = point.reshape(-1)
        x1, x2, x3, x4 = x1 * w, x2 * w, x3 * w, x4 * w
        y1, y2, y3, y4 = y1 * h, y2 * h, y3 * h, y4 * h
        src = numpy.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype="float32")
        dst = numpy.array([[0, 0], [144, 0], [0, 48], [144, 48]], dtype="float32")
        box = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
        M = cv2.getPerspectiveTransform(src, dst)
        out_img = cv2.warpPerspective(image, M, (144, 48))
        return out_img, box


if __name__ == '__main__':
    # image = cv2.imread('test_image.jpg')
    # boxes = read_plate(image)
    # print(boxes)
    images = []
    read_plate = ReadPlate()
    for file_name in os.listdir('D:/dataSet/CCPD2019/5000_images/test'):
        # for image_name in os.listdir(f'/home/cq/public/hibiki/CCPD2019/ccpd_db/{file_name}'):
        images.append(f'D:/dataSet/CCPD2019/5000_images/test/{file_name}')
    yes, yesss, count = 0, 0, 0
    s = time.time()
    for path in images:

        label_number = path.split('/')[-1].split('-')[4].split('_')
        label = ''
        for i , c in enumerate(label_number): #车牌翻译
            if i == 0:
                label += config.class_name_ccpd_chi[int(c)]
            else:
                label += config.class_name_ccpd_num[int(c)]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        count += 1
        results = read_plate(image)
        for plate, box in results:
            i+=1
            print(yes, yesss, count, yes / count, yesss / count, label, plate)
            if label == plate:
                yes += 1
            if label[1:] == plate[1:]:
                yesss += 1
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            image = DrawChinese(image, plate, (int(box[0]), int(box[1]) - 50), 40, (200, 0, 0))
            #cv2.imshow('a', image)
            #cv2.waitKey(0)
            break

    print(yes, count, yes / count, round(time.time()-s))