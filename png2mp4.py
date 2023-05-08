import cv2
import os

# 读取第一帧，获取宽高信息
path = "video.mp4"
filename = "frame1.png"
img = cv2.imread(os.path.join(path, filename))
height, width, layers = img.shape

# 设置视频编码器和输出文件名
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("output.mp4", fourcc, 120, (width, height))

# 逐帧读取图片，写入视频文件
for i in range(1, 481):
    filename = "frame" + str(i) + ".png"
    img = cv2.imread(os.path.join(path, filename))
    video.write(img)

# 释放资源
cv2.destroyAllWindows()
video.release()

