# yolo格式坐标转化为voc格式
def coordinateYolo2voc(size, box):
    width = size[0]
    height = size[1]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x_center = float(x) * width
    y_center = float(y) * height
    xminVal = int(x_center - 0.5 * float(w) * width)
    yminVal = int(y_center - 0.5 * float(h) * height)
    xmaxVal = int(x_center + 0.5 * float(w) * width)
    ymaxVal = int(y_center + 0.5 * float(h) * height)

    list = [xminVal, yminVal, xmaxVal, ymaxVal]

    return list


# voc格式坐标转化为yolo格式
def coordinateVoc2yolo(size, box):
    # xmin,ymin,xmax,ymax
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]

    list = [x, y, w, h]
    return list


# 日志info转换
def conver_log(message, para, para_val):
    info = str(message) + ","
    for i in range(len(para)):
        info = info + str(para[i]) + ":[" + str(para_val[i]) + "]"
    return info
