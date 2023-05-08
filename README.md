# YOLOv8_ByteTrack_Honeybee
## 本科毕业设计记录
### 特性
- 基于YOLOv8目标检测网络和ByteTrack多目标跟踪器部署实现
- 引入特征金字塔P2层的细节位置信息并并删除Head部分的顶层分支
- 利用类间最大方差划分高低分框，利用precision和recall线性修正最大存活帧数和IoU关联阈值
### 演示
![Alt Text](track.gif)

