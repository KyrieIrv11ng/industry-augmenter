import numpy as np


# 计算RGB三通道均值与方差
def compute_RGB_mean_var(img):
    RGB_mean = []
    RGB_var = []
    # print(img[:, :, 0])
    # print(img[:, :, 1])
    # print(img[:, :, 2])
    R_mean = np.mean(img[:, :, 2])
    G_mean = np.mean(img[:, :, 0])
    B_mean = np.mean(img[:, :, 1])
    RGB_mean.append(R_mean)
    RGB_mean.append(G_mean)
    RGB_mean.append(B_mean)
    # print(RGB_mean)
    R_var = np.var(img[:, :, 2])
    G_var = np.var(img[:, :, 0])
    B_var = np.var(img[:, :, 1])
    RGB_var.append(R_var)
    RGB_var.append(G_var)
    RGB_var.append(B_var)
    # print(RGB_var)
    return RGB_mean, RGB_var


# 计算角度
def rad(x):
    return x * np.pi / 180


# 计算标注
def anno_compute(large_size, small_size, box):
    large_width = large_size[0]
    large_height = large_size[1]
    small_width = small_size[0]
    small_height = small_size[1]
    new_w = small_width / large_width
    new_h = small_height / large_height
    box = [box[0], box[1], new_w, new_h]
    return box
