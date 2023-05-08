'''
jpg2mp4:将jpg序列的格式转为mp4格式输出
'''
import ffmpeg

# 设置文件名模板
input_filename = 'source\\%06d.jpg'

# 设置输出文件名
output_filename = 'output.mp4'

# 设置帧率
framerate = 60

# 创建FFmpeg命令
(
    ffmpeg
    .input(input_filename, framerate=framerate)
    .output(output_filename)
    .run()
)
