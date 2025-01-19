import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import re

MODEL_NAMES = ["Yolov5Lite-s", "Yolov5s"]
COLORS = ["#e64ae4", "#37d656", "#d63759"]


# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="读取多个FPS文件并绘制折线图")
    parser.add_argument('--files', metavar='F', type=str, nargs='+', help='一个或多个FPS数据文件')
    return parser.parse_args()


# 读取文件中的FPS数据
def read_fps_data(file_name):
    fps_data = []
    with open(file_name, 'r') as file:
        fps_data = [float(line.strip()) for line in file.readlines()]
    return fps_data


# 绘制折线图
def plot_fps(files):
    plt.figure(figsize=(10, 6))

    # 遍历每个文件，读取数据并绘制
    for idx in range(len(files)):
        file_name = files[idx]
        fps_data = read_fps_data(file_name)

        # 对数据进行插值，使其平滑
        x = np.arange(len(fps_data))  # 原始的 x 数据（索引）
        # x_new = np.linspace(x.min(), x.max(), 500)  # 新的 x 数据（更细的间隔）
        # spl = make_interp_spline(x, fps_data, k=3)  # 使用三次样条插值
        # y_new = spl(x_new)  # 计算插值后的 y 数据

        mean_value = np.mean(fps_data)
        # model_name = MODEL_NAMES[idx]
        model_name = file_name
        plt.plot(x, fps_data, color=COLORS[idx], label=model_name)  # 绘制平滑后的折线图
        plt.axhline(mean_value, color=COLORS[idx], linestyle='--', label=f'{model_name} mean FPS: {mean_value:.2f}')

    plt.ylim(0, 1.5)
    plt.title('FPS Comparison')
    plt.xlabel('Time (seconds)')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True)
    plt.show()


# 主函数
if __name__ == '__main__':
    
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 18
    # plt.rcParams['axes.linewidth'] = 2

    args = parse_args()  # 解析命令行参数
    plot_fps(args.files)  # 绘制多个文件的FPS折线图
