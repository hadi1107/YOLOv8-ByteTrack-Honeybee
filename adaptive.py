"""
adaptive：自适应检测结果的跟踪参数设置方法实现
"""
import numpy as np

def calculate_iou_threshold(precision,base = 0.75,alpha = 0.1):
    t_base = base
    a = alpha
    t_iou = t_base + a * precision
    return t_iou

def calculate_age_start(recall,base = 30,alpha = 20):
    start_base = base
    age_base = base
    a = alpha
    b = alpha
    start = start_base - a * (1-recall)
    age = age_base + b * (1-recall)
    return age,start

def calculate_otsu_threshold(confidences,bins = 100):
    # 将输入列表转换为NumPy数组
    confidences = np.array(confidences)

    # 将置信度等级设置为100
    num_bins = 100

    # 在0到1之间创建num_bins个均匀分布的区间
    bin_edges = np.linspace(0, 1, num_bins+1)

    # 计算输入数据的直方图
    hist, _ = np.histogram(confidences, bins=bin_edges)

    # 计算每个区间的中心值
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 计算所有数据的总和以及它们乘以对应的中心值的总和
    total_confidences = np.sum(hist)
    print("检测框个数:",total_confidences)
    total_sum = np.sum(np.multiply(hist,bin_centers))
    print("检测框置信度总和:",total_sum)

    # 初始化最大方差和阈值变量
    max_variance = 0
    threshold = 0

    # 初始化背景类的置信度总和和置信度数量
    sumB = 0
    wB = 0

    # 循环遍历每个置信度
    for i in range(num_bins):
        # 将当前区间的像素数添加到背景类的像素数量中
        wB += hist[i]
        # 如果背景类的像素数为0，则跳过该灰度级
        if wB == 0:
            continue

        # 计算前景类的像素数
        wF = total_confidences - wB
        # 如果前景类的像素数为0，则跳出循环
        if wF == 0:
            break

        # 计算背景类的平均像素值和前景类的平均像素值
        sumB += hist[i] * bin_centers[i]
        mB = sumB / wB
        mF = (total_sum - sumB) / wF

        # 计算背景类和前景类之间的方差
        var_between = wB * wF * (mB - mF) ** 2

        # 如果当前方差大于最大方差，则更新最大方差和阈值
        if var_between > max_variance:
            max_variance = var_between
            threshold = bin_centers[i]

    # 返回最大方差对应的阈值
    return threshold

if __name__ == "__main__":
    # 示例：获取置信度集合
    # confidences = get_confidences()
    # precision = get_precision()
    # recall = get_recall()
    confidences = [0.7,0.69,0.56,0.42,0.69,0.75,0.72,
                   0.71,0.38,0.75,0.59,0.67,0.57,0.66,
                   0.66,0.69,0.73,0.72,0.75,0.58,0.74,
                   0.70,0.38,0.35,0.27,0.76,0.71,0.74,
                   0.44,0.59,0.63,0.63,0.69,0.46,0.68,
                   0.38,0.50,0.54,0.73,0.53,0.54,0.64,
                   0.50,0.54,0.32,0.37,0.65,0.70,0.59]
    precision = 0.778
    recall = 0.96

    # 计算高低分框阈值
    threshold = calculate_otsu_threshold(confidences)
    iou_threshold = calculate_iou_threshold(precision)
    age,start = calculate_age_start(recall)

    print("最佳高低分框阈值:", threshold)
    print("最佳IOU阈值:", iou_threshold)
    print("最佳age阈值:", age)
    print("最佳start阈值:", start)



