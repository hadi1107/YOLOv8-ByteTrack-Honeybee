from ultralytics import YOLO
import os
import numpy as np
import random
seed = 3407
np.random.seed(seed) # seed是一个固定的整数即可
random.seed(seed)

if __name__ == '__main__':
    #解决freeze的bug
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    #加载模型
    model = YOLO("best.pt")  # 加载模型
    #验证
    model.info()
    results = model("photo.jpg",save = True,device = '0')
    #metrics = model.val()  # 在验证集上评估模型性能