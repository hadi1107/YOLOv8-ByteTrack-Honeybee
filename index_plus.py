"""
indexplus：将Honeybee2d的000000起始的帧转为000001起始的帧
"""
import os

# 源文件夹和目标文件夹
src_folder = "source"
dst_folder = "destination"

# 文件名模板，{}会被替换为数字
src_template = "000{:03d}.jpg"
dst_template = "{:06d}.jpg"
file_names = os.listdir(src_folder)

# 遍历源文件夹中的所有文件
for i in range(len(file_names)):
    # 构造源文件名和目标文件名
    src_name = os.path.join(src_folder, src_template.format(i))
    dst_name = os.path.join(dst_folder, dst_template.format(i+1))

    # 移动文件
    os.rename(src_name, dst_name)
