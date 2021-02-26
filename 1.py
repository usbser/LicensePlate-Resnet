
import easyocr
import os
import cv2
from time import time
reader = easyocr.Reader(['ch_sim', 'en'])
for file_name in os.listdir('/home/cq/public/hibiki/2/test'):
    path = f'/home/cq/public/hibiki/2/test/{file_name}'
    # reader = easyocr.Reader(['ch_sim', 'en'])
    s = time()
    result = reader.readtext(path)
    print(time()-s)
    print(result)
    image = cv2.imread(path)
    image = cv2.resize(image,None,fx=0.3,fy=0.3)
    cv2.imshow('a',image)
    cv2.waitKey()



