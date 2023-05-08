"""
sort：利用卡尔曼滤波和匈牙利匹配处理检测结果，形成跟踪轨迹，并可视化保存结果
"""
# __future__ 模块允许在 Python 2.x 中使用 Python 3.x 特性
from __future__ import print_function

# 导入所需库
import os
import numpy as np
import matplotlib
# 设置matplotlib的后端为TkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 用于读取和显示图像的模块
from skimage import io
# 用于在文件系统中查找文件路径名的模块
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

# 为numpy随机数生成器设置种子,确保结果的可重复性
np.random.seed(0)

# 定义线性分配函数，用于解决二分图最小权匹配问题（匈牙利匹配）
def linear_assignment(cost_matrix):
    try:
        # 导入lap库并尝试使用Jonker-Volgenant算法
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        # 如果i >= 0，返回匹配结果的数组
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        # 如果无法导入lap库，使用scipy库中的linear_sum_assignment函数
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        # 返回匹配结果的数组
        return np.array(list(zip(x, y)))

# 定义计算两组边界框之间的IOU的函数
def iou_batch(bbox_1, bbox_2):
    """
    从SORT中获取：计算两个边界框之间的IOU，输入形式为[x1,y1,x2,y2]
    """
    bbox_2 = np.expand_dims(bbox_2, 0)
    bbox_1 = np.expand_dims(bbox_1, 1)

    # 计算两组边界框相交矩形的坐标
    xx1 = np.maximum(bbox_1[..., 0], bbox_2[..., 0])
    yy1 = np.maximum(bbox_1[..., 1], bbox_2[..., 1])
    xx2 = np.minimum(bbox_1[..., 2], bbox_2[..., 2])
    yy2 = np.minimum(bbox_1[..., 3], bbox_2[..., 3])

    # 计算相交矩形的宽度和高度
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    # 计算相交矩形的面积
    wh = w * h
    # 计算IOU
    iou = wh / ((bbox_1[..., 2] - bbox_1[..., 0]) * (bbox_1[..., 3] - bbox_1[..., 1])
        + (bbox_2[..., 2] - bbox_2[..., 0]) * (bbox_2[..., 3] - bbox_2[..., 1]) - wh)
    return iou

# 将边界框从[x1,y1,x2,y2]形式转换为[x,y,s,r]形式
def convert_bbox_to_z(bbox):
    """
    输入边界框形式为[x1,y1,x2,y2]，返回z形式为[x,y,s,r]，其中x,y为边界框中心，
    s为边界框的面积，r为宽高比
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale即面积
    r = w / float(h)  # 宽高比
    return np.array([x, y, s, r]).reshape((4, 1))

# 将边界框从[x,y,s,r]形式转换为[x1,y1,x2,y2]形式
def convert_x_to_bbox(x, score=None):
    """
    输入边界框形式为[x,y,s,r]，返回边界框形式为[x1,y1,x2,y2]，其中x1,y1为左上角坐标，
    x2,y2为右下角坐标
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    这个类表示单个跟踪对象的内部状态，对象以边界框（bbox）的形式观察到。
    """
    count = 0
    def __init__(self,bbox):
        """
        使用初始边界框初始化跟踪器.
        """
        #定义恒速模型，kf是一个卡尔曼滤波器对象
        #dim_x 被设置为 7，表示状态向量包含目标位置、速度和加速度的 x 和 y 坐标以及目标 ID。dim_z 被设置为 4，表示测量向量包含检测器提供的目标位置的 x 和 y 坐标
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        #F为状态转移矩阵
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        #H为观测矩阵
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        #R为观测噪声
        self.kf.R[2:,2:] *= 10.
        #P为协方差矩阵
        self.kf.P[4:,4:] *= 1000.
        #给不可观察的初始速度赋予高不确定性
        self.kf.P *= 10.
        #Q为过程噪声
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        #将目标检测器检测到的边界框（bbox）转换成卡尔曼滤波器需要的状态向量x的前四个元素（位置和速度）
        #𝑢, 𝑣, 𝛾, ℎ分别为边界框中心的横坐标和纵坐标、纵横比和宽度
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        #KalmanBoxTracker.count变量是一个类级别的变量（类static成员），每次创建新实例时都会递增。
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        使用观察到的边界框bbox更新状态向量。
        """
        self.time_since_update = 0  # 时间自上次更新以来的计数器重置为0
        self.history = []  # 历史记录清空
        self.hits += 1  # 增加“命中”次数
        self.hit_streak += 1  # 增加连续“命中”次数
        self.kf.update(convert_bbox_to_z(bbox))  # 使用边界框bbox将卡尔曼滤波器KF的状态向量进行更新

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计。
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):  # 如果状态向量中的第7个元素与第3个元素的和小于等于0
            self.kf.x[6] *= 0.0  # 将第7个元素设置为0
        self.kf.predict()  # 将状态向量推进一步
        self.age += 1  # 增加年龄计数器
        if (self.time_since_update > 0):  # 如果自上次更新以来的时间计数器大于0
            self.hit_streak = 0  # 将连续“命中”计数器重置为0
        self.time_since_update += 1  # 将更新后的时间计数器加1
        self.history.append(convert_x_to_bbox(self.kf.x))  # 将推进后的状态向量转换为边界框，并将其添加到历史记录中
        return self.history[-1]  # 返回最后一个历史记录的边界框

    def get_state(self):
        """
        返回当前的边界框估计。
        """
        return convert_x_to_bbox(self.kf.x)  # 将当前状态向量转换为边界框并返回

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    将检测到的物体与已经跟踪的物体进行关联，两者都表示为边界框。
    返回3个列表：匹配、未匹配检测、未匹配跟踪器
    """
    if len(trackers) == 0:
        # 如果没有跟踪器，将所有检测都标记为未匹配的检测
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # 计算IoU矩阵
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        # 如果IoU矩阵的形状都大于0，则通过阈值将匹配的物体对筛选出来
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # 如果有且仅有一个检测与一个跟踪器匹配，则直接选择这对匹配
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 否则，使用匈牙利算法进行关联
            matched_indices = linear_assignment(-iou_matrix)
    else:
        # 如果IoU矩阵的形状都为0，则说明没有匹配，返回空列表
        matched_indices = np.empty(shape=(0, 2))

    # 将未匹配的检测和跟踪器分别存储在列表中
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # 将IoU值小于阈值的匹配去除，标记为未匹配的检测和跟踪器
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # 返回匹配、未匹配检测和未匹配跟踪器
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

#Sort类，接受三个初始化参数
class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        初始化SORT的关键参数
        """
        self.max_age = max_age  # 最大跟踪次数
        self.min_hits = min_hits  # 最小命中次数
        self.iou_threshold = iou_threshold  # IoU阈值
        self.trackers = []  # 跟踪器列表
        self.frame_count = 0  # 帧计数器

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            dets - 以[[x1，y1，x2，y2，score]，[x1，y1，x2，y2，score]，...]格式提供的numpy数组，类似bbox
        Requires:
            无论有无检测结果，都必须针对每个帧调用此方法（对于没有检测到的帧，请使用np.empty((0, 5))）。
            返回一个类似的数组，其中最后一列是物体ID。
        Note：
            返回的物体数可能与提供的检测数不同。
        """
        self.frame_count += 1
        # 获取现有跟踪器的预测位置，trks是一个trackers.size()行,5列的二维np数组
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        # 在这个 for 循环中，我们使用 enumerate() 函数来获取数组 trks 的索引和元素。在每次迭代中，t 变量被设置为当前元素的索引，trk 变量被设置为当前元素的值。
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # 获取当前跟踪器的预测位置
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]  # 更新轨迹数组中对应跟踪器的位置信息
            if np.any(np.isnan(pos)):  # 如果预测位置包含 NaN，则将其添加到删除列表中
                to_del.append(t)

        # 从轨迹数组中删除包含 NaN 位置的跟踪器，以及在删除列表中的跟踪器
        # np.ma.masked_invalid() 函数创建一个掩码数组，用于标记轨迹数组中所有包含 NaN 值的行。
        # np.ma.compress_rows() 函数对轨迹数组进行压缩，将所有包含 NaN 值的行从数组中删除。
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 使用匈牙利算法将当前帧的检测结果与跟踪器关联起来
        # 返回值包括：
        # matched：已经匹配的检测结果与跟踪器的对应列表
        # unmatched_dets：未匹配的检测结果列表
        # unmatched_trks：未匹配的跟踪器列表
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 更新已匹配的跟踪器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 为未匹配的检测创建并初始化新的跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # 删除轨迹
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

def parse_args():
    """
    解析输入参数。
    """
    # 创建一个解析器对象，它用于解析命令行参数,description提供简要描述。
    parser = argparse.ArgumentParser(description='SORT demo')
    # 显示跟踪结果（速度较慢）,dest='display' 意味着将该参数的值存储在 display 变量中
    # 如果用户在命令行中输入了 --display，则将 display 变量的值设置为 True
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
    # 检测结果路径,默认为data
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    # seq_path中的子目录，默认为data\train，这个子目录存放det\det.txt数据
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='mytrain')
    # 最大无检测跟踪帧数,作者的default == 1
    parser.add_argument("--max_age", help="Maximum number of frames to keep alive a track without associated detections.", type=int, default=2)
    # 最小匹配次数才能初始化跟踪器，作者的default == 3
    parser.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.", type=int, default=2)
    # 匹配的最小IOU值,作者的default == 0.3
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.2)
    args = parser.parse_args()
    return args

#main入口
if __name__ == '__main__':
    # 获取命令行参数
    args = parse_args()
    # 是否显示跟踪结果
    display = args.display
    # seq_path中的子目录,自定义数据集时，使用--phase mytrain
    phase = args.phase
    # 总跟踪时间
    total_time = 0.0
    # 总跟踪帧数
    total_frames = 0
    # 随机颜色，仅用于显示
    colours = np.random.rand(64, 3)
    if display:
        # 如果需要显示跟踪结果，需要链接到 MOT benchmark 数据集 ，自定义数据集则在该文件夹下放置mytrain
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n')
            exit()
        #打开plt的交互模式
        plt.ion()
        #创建空窗口
        fig = plt.figure()
        #创建一个子图
        ax1 = fig.add_subplot(111, aspect='equal')
    # 创建 output 文件夹，用于保存跟踪结果
    if not os.path.exists('output'):
        os.makedirs('output')
    # 使用通配符匹配phase目录下的所有文件，pattern 形如 data/train/*/det/det.txt
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # 处理每个跟踪序列,seq_dets_fn即每个det.txt
    for seq_dets_fn in glob.glob(pattern):
        # 创建一个SORT跟踪器实例
        mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
        # 读取跟踪序列的检测结果
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        # 获取跟踪序列名称seq
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        if seq == "HoneyBee-05_120" or seq == "HoneyBee-05_60":
            continue
        # 用获取到的seq打开输出文件,单独给每个seq一个文件夹
        output_dir = os.path.join('output', seq + '_output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, '%s.txt'%(seq)),'w') as out_file:
            print("Processing seq:%s."%(seq))
            #定义两个字典，负责保存对应id的跟踪框中心点
            x_points = {}
            y_points = {}
            # 遍历每一帧
            for frame in range(int(seq_dets[:,0].max())):
                frame += 1
                # 获取当前帧的检测结果,seq_dets[:, 0] == frame 表示匹配当前帧的det行，2：7表示左上角xy和宽高和score
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                # 将检测结果转换为bbox的形式，从而输入给tracker
                dets[:, 2:4] += dets[:, 0:2]
                # 总帧数+1
                total_frames += 1
                if display:
                    # 加载当前帧的图像，格式为帧数（6位数）.jgp
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
                    im =io.imread(fn)
                    # 加载背景，和标注框一起注释可以得到纯净轨迹
                    ax1.imshow(im)
                    plt.title(seq)
                # 开始跟踪
                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    #输出格式：帧号、ID、x、y、x2-x1、y2-y1
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
                    if(display):
                        #跟踪id可视化
                        d = d.astype(np.int32)
                        #保存当前跟踪框的跟踪点
                        center_x = (d[0]+d[2]) // 2
                        center_y = (d[1]+d[3]) // 2
                        #为每个id分配一个list保存中心点集，作为跟踪轨迹可视化出来
                        if d[4] not in x_points:
                            x_points[d[4]] = []
                        x_points[d[4]].append(center_x)
                        if d[4] not in y_points:
                            y_points[d[4]] = []
                        y_points[d[4]].append(center_y)
                        #为每个id分配一个随机颜色
                        rcolor = colours[d[4]%64,:]
                        #标记当前跟踪框
                        #ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=2,ec=rcolor))
                        #标记当前跟踪框的目标id
                        ax1.text(d[0], d[1], "ID=" + str(d[4]), fontsize=6, color=rcolor)
                        #绘制当前还在跟踪中的目标轨迹
                        ax1.plot(x_points[d[4]],y_points[d[4]], 'o', color=rcolor, markersize=1)

                #按帧显示当前绘制并清空当前绘制
                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    #保存当前跟踪图片
                    fig.savefig(os.path.join(output_dir, 'frame{}.png'.format(frame)),dpi = 300)
                    ax1.cla()
            print("seq:"+seq+" Total Tracking took: %.3f seconds for %d frames or %.1f FPS or %.6f s/frame" % (total_time, total_frames, total_frames / total_time,total_time/total_frames))


    #输出当前模式的效率值
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS or %.6f s/frame" % (total_time, total_frames, total_frames / total_time,total_time/total_frames))

    #用到plt可视化的太慢，不用plt来给出纯计算消耗的时间作为性能比较
    if(display):
        print("Note: to get real runtime results run without the option: --display")
