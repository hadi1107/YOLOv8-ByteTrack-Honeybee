'''
track:YOLOv8的跟踪主函数
'''
from ultralytics import YOLO
import os
import random
import numpy as np
seed = 3407
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)

if __name__ == '__main__':
    #解决freeze的bug
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    #加载模型
    model = YOLO("weights\\best.pt")  # 加载模型
    model.info()
    #跟踪
    results = model.track("video8s.mp4",tracker="bytetrack.yaml",save = True,device = '0')
