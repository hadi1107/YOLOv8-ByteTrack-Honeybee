"""
json2yolo：将Honeybee2d的json转为适合yolo的txt格式
"""
import json

for i in range(480):
    # 读取JSON文件
    with open(f'annotation/{i:06}.json', 'r') as f:
        data = json.load(f)

    # 解析JSON文件并处理结果
    results = []
    for shape in data['shapes']:
        # 提取label、points、width、height信息
        points = shape['points']
        x = points[1][0] + points[0][0]
        x /= 2
        y = points[1][1] + points[0][1]
        y /= 2
        w = points[1][0] - points[0][0]
        h = points[1][1] - points[0][1]
        # 归一化坐标和尺寸到[0, 1]范围内
        x /= 1920
        y /= 1080
        w /= 1920
        h /= 1080
        # 将矩形信息添加到结果列表中
        results.append([0, abs(x), abs(y), abs(w), abs(h)])

    # 将结果保存到txt文件中
    with open(f'labels/{i:06}.txt', 'w') as f:
        for line in results:
            f.write(' '.join(map(str, line)) + '\n')
