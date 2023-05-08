"""
json2det：将Honeybee2d的json转为适合sort的txt格式
"""
import json
import numpy as np

# 读取 json 文件中的数据
with open('data.json', 'r') as f:
    data = json.load(f)

# 获取 annotations 中所有 bbox 和 confidence
bboxes = []
confidences = []
image_ids = []
for annotation in data['annotations']:
    bboxes.append(annotation['bbox'])
    confidences.append(annotation['confidence'])
    image_ids.append(annotation['image_id'])

# 转换为 numpy 数组
bboxes = np.array(bboxes)
confidences = np.array(confidences)
image_ids = np.array(image_ids)

# 构造 det 格式的数组
num_annotations = len(data['annotations'])
det_array = np.zeros((num_annotations, 10))
det_array[:, 0] = image_ids
det_array[:, 1] = -1
det_array[:, 2] = bboxes[:, 0]
det_array[:, 3] = bboxes[:, 1]
det_array[:, 4] = bboxes[:, 2] - bboxes[:, 0]
det_array[:, 5] = bboxes[:, 3] - bboxes[:, 1]
det_array[:, 6] = confidences
det_array[:, 7] = -1
det_array[:, 8] = -1
det_array[:, 9] = -1

# 保存为 det 文件
np.savetxt('det.txt', det_array, fmt='%.0f, %.0f, %.2f, %.2f, %.2f, %.2f, %.6f, %.0f, %.0f, %.0f ')
