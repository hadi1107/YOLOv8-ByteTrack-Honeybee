'''
make_list:输出适用于YOLO的训练格式txt文件，即对应图像/标签的绝对路径
'''
import os

# 训练和验证文件夹路径
train_dir = 'bees480/images/train'
train_ann_dir = 'bees480/labels/train'
val_dir = 'bees480/images/val'
val_ann_dir = 'bees480/labels/val'

img_ext = '.jpg'
ann_ext = '.txt'

# 生成train.txt文件
train_txt = open('train.txt', 'w')
for img_file in os.listdir(train_dir):
    if img_file.endswith(img_ext):
        img_path = os.path.join(train_dir, img_file)
        ann_file = os.path.splitext(img_file)[0] + ann_ext
        ann_path = os.path.join(train_ann_dir, ann_file)
        if os.path.exists(ann_path):
            train_txt.write(img_path + '\n')
train_txt.close()

# 生成val.txt文件
val_txt = open('val.txt', 'w')
for img_file in os.listdir(val_dir):
    if img_file.endswith(img_ext):
        img_path = os.path.join(val_dir, img_file)
        ann_file = os.path.splitext(img_file)[0] + ann_ext
        ann_path = os.path.join(val_ann_dir, ann_file)
        if os.path.exists(ann_path):
            val_txt.write(img_path + '\n')
val_txt.close()
