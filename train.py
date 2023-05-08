'''
train:YOLOv8的训练主函数
'''
from ultralytics import YOLO
import os
import random
import numpy as np
seed = 3407
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)

if __name__ == '__main__':
    # 解决freeze的bug
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # 加载模型
    model = YOLO("yolov8s-p2.yaml")  # 构建新模型
    exit()
    #model = YOLO("yolov8s.pt")  # 加载预训练模型用于训练
    # 训练模型
    results = model.train(data="bees480.yaml", epochs=100, workers = 1, batch = 8,device='0',optimizer = 'Adam',lr0 = 0.001)
    # 在验证集上评估模型性能
    results = model.val()
