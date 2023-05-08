'''
show_det:可视化yolo格式的标记，便于验证数据集的适用性
'''
import cv2

# 加载图像
img = cv2.imread("photo.jpg")
img_height, img_width, _ = img.shape

# 解析txt文件
with open("label.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().split(" ")
    # 解析位置和大小信息
    class_id, x_center, y_center, width, height = line
    # 转换为实际坐标和像素大小
    x_min = int((float(x_center) - float(width) / 2) * 1920)
    y_min = int((float(y_center) - float(height) / 2) * 1080)
    x_max = int((float(x_center) + float(width) / 2) * 1920)
    y_max = int((float(y_center) + float(height) / 2) * 1080)

    # 绘制检测框
    color = (0, 255, 0) # 绿色
    thickness = 2 # 线条粗细
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

# 显示图像
cv2.imshow("image", img)
cv2.waitKey(0)

# 保存图像
cv2.imwrite("image_with_detection.jpg", img)
