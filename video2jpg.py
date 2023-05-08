'''
video2jpg:将Track输出的mp4格式视频转为jpg序列的格式
'''
import cv2
import os

# 设置视频文件名
input_filename = 'runs\\best.mp4'

# 创建一个VideoCapture对象来读取视频
cap = cv2.VideoCapture(input_filename)

# 检查是否成功打开视频
if not cap.isOpened():
    print('Failed to open video file:', input_filename)
    exit()

# 设置输出文件夹
output_folder = 'jpgresult'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取并保存每一帧图像
frame_index = 0
while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 检查是否读取成功
    if not ret:
        break

    # 生成输出文件名
    output_filename = os.path.join(output_folder, '{:06d}.jpg'.format(frame_index))

    # 保存图像
    cv2.imwrite(output_filename, frame)

    # 更新帧计数器
    frame_index += 1

# 释放VideoCapture对象
cap.release()

print('Finished converting MP4 to JPG images:', frame_index, 'frames saved.')
