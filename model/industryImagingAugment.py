from common.utils.compute import *
from common.utils.conver import *
from common.utils.log import *
import cv2
import numpy as np
import math
import random


class IndustryImagingAugment():
    # 视角变换模型
    # ----------------------------------------------------------------------------------------
    # 参数控制：
    # img：工件图像；类型：img
    # box：缺陷位置中心点坐标与尺寸；类型：长度为4的list列表；示例：[0.5,0.5,0.1，0.1]
    #       box[0]：缺陷所处位置中心点x值，x坐标/工件宽度width，double类型
    #       box[1]：缺陷所处位置中心点y值，y坐标/工件高度height，double类型
    #       box[2]：缺陷宽度w，w/工件宽度weight，double类型
    #       box[2]：缺陷宽度h，h/工件宽高度height，double类型
    # anglex：绕x轴旋转角度；取值范围：[0,70]；类型：int；示例：45
    # angley：绕y轴旋转角度；取值范围：[0,70]；类型：int；示例：45
    # anglez：绕z轴旋转角度；取值范围：[0,360]；类型：int；示例：45
    # s：缩放倍数；取值范围：（0.5,2]；类型：double；示例：1.5
    # 注：
    # 灰度图像色调均保持为0
    # 如果是三通道图像自行在范围内取任意值或随机值
    def perspectiveTrans(self, img, objects, anglex=None, angley=None, anglez=None, H=None):
        log = Log()
        # 参数控制，None值参数取随机值
        if (anglex is None):
            anglex = random.randint(0, 70)
        if (angley is None):
            angley = random.randint(0, 70)
        if (anglez is None):
            anglez = random.randint(0, 360)
        if (H is None):
            H = random.randint(-1000, 1000)

        # 前置校验
        if (anglex < 0 or anglex > 70):
            assert "俯仰角调节参数范围应在[0,70]"
        if (angley < 0 or angley > 70):
            assert "横滚角调节参数范围应在[0,70]"
        if (anglez < 0 or anglez > 360):
            assert "航向角调节参数范围[0,360]"
        if (H < -0.5 or H > 2.0):
            assert "高度参数范围应在[-500,500]"

        # 打日志
        log.info(conver_log("工业图像采集过程模拟视角变换模型", ["俯仰角调节参数", "横滚角调节参数", "航向角调节参数", "高度参数"],
                            [anglex, angley, anglez, H]))
        log.info(conver_log("视角变换前工件图像尺寸", ["工件宽度width", "工件高度height"],
                            [img.shape[1], img.shape[0]]))
        for object in objects:
            log.info(conver_log("视角变换前缺陷中心点坐标与尺寸", ["缺陷中心点坐标x", "缺陷中心点坐标y", "缺陷宽度width", "缺陷高度height"],
                                [img.shape[1] * float(object[1]), img.shape[0] * float(object[2]),
                                 img.shape[1] * float(object[3]),
                                 img.shape[0] * float(object[4])]))

        # 扩展图像，保证内容不超出可视范围
        img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
        h, w = img.shape[0:2]
        size = [w, h]
        fov = 21

        # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

        z_old = z
        # 高度变换
        z = z + H

        # 如果变换小于0，重新生成
        if (z < 0):
            z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
            log.info("相机高度H变换后小于0，不做变换处理")
        log.info("视角高度变换前相机高度为：" + str(z_old))
        # 齐次变换矩阵
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                       [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                       [0, 1, 0, 0],
                       [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                       [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], np.float32)

        r = rx.dot(ry).dot(rz)

        # 四对点的生成
        pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, h, 0, 0], np.float32) - pcenter
        p4 = np.array([w, h, 0, 0], np.float32) - pcenter

        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)

        # print(dst1, dst2, dst3, dst4)

        list_dst = [dst1, dst2, dst3, dst4]

        org = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], np.float32)

        dst = np.zeros((4, 2), np.float32)

        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

        # print(dst)
        # 生成透视变换矩阵
        warpR = cv2.getPerspectiveTransform(org, dst)
        # print(warpR)
        # opencv透视变换
        result = cv2.warpPerspective(img, warpR, (h, w))

        height, width = result.shape[0:2]

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        xmin = width
        ymin = height
        xmax = 0
        ymax = 0
        # 裁剪掉多余的黑色背景
        for i in range(height):
            for j in range(width):
                if (gray[i, j] != 0):
                    xmin = min(xmin, j)
                    xmax = max(xmax, j)
                    ymin = min(ymin, i)
                    ymax = max(ymax, i)

        # 重新生成objects
        objects_new = []
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])
            # 标签变换（缺陷坐标点透视变换）
            defect_size = self.annotationTrans(size, box, anglex, angley, anglez, H)

            # 获取缺陷图像
            defect_img = result[defect_size[1]:defect_size[3], defect_size[0]:defect_size[2]]

            # 对缺陷图像进行二值化处理，缺陷位置像素值为0，非缺陷位置像素值为255
            bin_img = self.binarization(defect_img)

            # 裁剪掉缺陷多余部分，重新生成缺陷的四点坐标
            height_defect, width_defect = bin_img.shape[0:2]

            xmin_defect = width_defect
            ymin_defect = height_defect
            xmax_defect = 0
            ymax_defect = 0

            for i in range(height_defect):
                for j in range(width_defect):

                    if (bin_img[i, j] != 255):
                        xmin_defect = min(xmin_defect, j)
                        xmax_defect = max(xmax_defect, j)
                        ymin_defect = min(ymin_defect, i)
                        ymax_defect = max(ymax_defect, i)

            defect_size[0] = defect_size[0] + xmin_defect
            defect_size[1] = defect_size[1] + ymin_defect
            defect_size[2] = defect_size[0] + xmax_defect
            defect_size[3] = defect_size[1] + ymax_defect

            # 重新生成yolo格式的box
            x_center = (((defect_size[0] + defect_size[2]) / 2) - xmin) / (xmax - xmin)
            y_center = (((defect_size[1] + defect_size[3]) / 2) - ymin) / (ymax - ymin)
            w = (defect_size[2] - defect_size[0]) / (xmax - xmin)
            h = (defect_size[3] - defect_size[1]) / (ymax - ymin)

            new_box = [x_center, y_center, w, h]

            object_new = [object[0], x_center, y_center, w, h]

            objects_new.append(object_new)

        # 裁剪最后得到的图像
        result = result[ymin:ymax, xmin:xmax]

        result = self.biLinearInterpolation(result,(ymax - ymin)*z/z_old,(xmax - xmin)*z/z_old)

        # 变换后打日志
        log.info(conver_log("视角变换后工件图像尺寸", ["工件宽度width", "工件高度height"],
                            [result.shape[1], result.shape[0]]))
        log.info("视角高度变换后相机高度为：" + str(z))


        for object in objects_new:
            log.info(conver_log("视角变换后缺陷中心点坐标与尺寸", ["缺陷中心点坐标x", "缺陷中心点坐标y", "缺陷宽度width", "缺陷高度height"],
                                [result.shape[1] * object[1], result.shape[0] * object[2], result.shape[1] * object[3],
                                 result.shape[0] * object[4]]))


        # 缩放所变换的标注坐标不需要重新变换
        return result, objects_new

    # 色彩变换模型
    # ----------------------------------------------------------------------------------------
    # 参数控制：
    # defect_img：缺陷图像；类型：img
    # light：亮度值；取值范围：（-1,1）；类型：double；示例：0,5
    # saturatuin：对比度值；取值范围：（-1,1）；类型：double；示例：0.5
    # tone_r：r通道色调值；取值范围：（-1,1）；类型：double；示例：0.5
    # tone_g：g通道色调值；取值范围：（-1,1）；类型：double；示例：0.5
    # tone_b：b通道色调值；取值范围：（-1,1）；类型：double：示例：0.5
    # 注：
    # 灰度图像色调均保持为0
    # 如果是三通道图像自行在范围内取任意值或随机值
    # 返回值：img
    def colorTrans(self, img, light=None, saturation=None, tone_r=None, tone_g=None, tone_b=None):
        log = Log()
        # 参数控制，None值参数取随机值
        if (light is None):
            light = random.uniform(-1, 1)
        if (saturation is None):
            saturation = random.uniform(-1, 1)
        if (tone_r is None and tone_g is None and tone_b is None):
            tone_r = tone_g = tone_b = random.uniform(-1, 1)

        # 前置校验
        if (light < -1 or light > 1):
            assert "亮度调节参数范围应在[-1,1]"
        if (saturation < -1 or saturation > 1):
            assert "饱和度调节参数范围应在[-1,1]"
        if (tone_b < -1 or tone_b > 1):
            assert "b通道色调调节参数范围应在[-1,1]"
        if (tone_g < -1 or tone_g > 1):
            assert "g通道色调调节参数范围应在[-1,1]"
        if (tone_r < -1 or tone_r > 1):
            assert "r通道色调调节参数范围应在[-1,1]"

        # 打日志
        log.info(conver_log("工业图像采集过程模拟色彩变换模型", ["亮度调节参数", "饱和度调节参数", "r通道色调调节参数", "g通道色调调节参数", "b通道色调调节参数"],
                            [light, saturation, tone_r, tone_g, tone_b]))

        # 获取图像RGB均值与方差
        # RGB_mean[0][1][2]--R均值，G均值，B均值
        # RGB_var[0][1][2]--R方差，G方差，B方差
        RGB_mean, RGB_var = compute_RGB_mean_var(img)
        R_mean, G_mean, B_mean = RGB_mean[0], RGB_mean[1], RGB_mean[2]
        R_var, G_var, B_var = RGB_var[0], RGB_var[1], RGB_var[2]

        rows, cols, channels = img.shape
        img_new = img.copy()
        # print(img)
        # 色彩变换公式
        for i in range(rows):
            for j in range(cols):
                R = img[i, j][2]
                B = img[i, j][1]
                G = img[i, j][0]

                # R通道变换
                R_mean_new = (1 + light) * R_mean + tone_r * R_mean
                R_var_new = (1 + saturation) * R_var
                R_new = (R - R_mean) / R_var * R_var_new + R_mean_new

                if R_new > 255:  # 防止像素值越界（0~255）
                    img_new[i, j][2] = 255
                elif R_new < 0:  # 防止像素值越界（0~255）
                    img_new[i, j][2] = 0
                img_new[i, j][2] = R_new

                # G通道变换
                G_mean_new = (1 + light) * G_mean + tone_g * G_mean
                G_var_new = (1 + saturation) * G_var
                G_new = (G - G_mean) / G_var * G_var_new + G_mean_new

                if G_new > 255:  # 防止像素值越界（0~255）
                    img_new[i, j][1] = 255
                elif G_new < 0:  # 防止像素值越界（0~255）
                    img_new[i, j][1] = 0
                img_new[i, j][1] = G_new

                # B通道变换
                B_mean_new = (1 + light) * B_mean + tone_b * B_mean
                B_var_new = (1 + saturation) * B_var
                B_new = (B - B_mean) / B_var * B_var_new + B_mean_new

                if B_new > 255:  # 防止像素值越界（0~255）
                    img_new[i, j][0] = 255
                elif B_new < 0:  # 防止像素值越界（0~255）
                    img_new[i, j][0] = 0
                img_new[i, j][0] = B_new
        # print(img_new)
        # cv2.imshow("1",img_new)
        # cv2.waitKey(0)
        return img_new

    # 分辨率变换模型
    # ----------------------------------------------------------------------------------------
    # 参数控制：
    # defect_img：缺陷图像；类型：img
    # s：缩放倍数；取值范围：（0.001,1）；类型：double；示例：2
    # 返回值：img
    def resolutionTrans(self, img, s=None):
        log = Log()
        # 参数控制
        if (s is None):
            s = random.uniform(0.6, 0.9)

        # 前置校验
        if (s < 0.6 or s > 0.9):
            assert "s调节参数范围应在[0.6,0.9]"

        # 打日志
        log.info(conver_log("工业图像采集过程模拟分辨率变换模型", ["缩放倍数"],
                            [s]))
        rows, cols, channels = img.shape
        img_large = self.biLinearInterpolation(img, rows * s, cols * s)
        rows, cols, channels = img_large.shape
        img_small = self.biLinearInterpolation(img_large, rows / s, cols / s)

        return img_small

    # ----------------------------------------------------------------------------------------
    # common_util
    # ----------------------------------------------------------------------------------------
    # 标注变换
    def annotationTrans(self, size, box, anglex, angley, anglez, H):
        width = size[0]
        height = size[1]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        fov = 21

        # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
        z = np.sqrt(width ** 2 + height ** 2) / 2 / np.tan(rad(fov / 2))

        z = z + H
        # print(z)
        # 齐次变换矩阵
        # 绕x轴旋转矩阵
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                       [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                       [0, 0, 0, 1]], np.float32)
        # 绕y轴旋转矩阵
        ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                       [0, 1, 0, 0],
                       [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                       [0, 0, 0, 1]], np.float32)
        # 绕z轴旋转矩阵
        rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                       [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], np.float32)
        # 整体旋转矩阵
        r = rx.dot(ry).dot(rz)
        # 转换坐标

        size_size = [size[0] - 400, size[1] - 400]
        list = coordinateYolo2voc(size_size, box)

        xmin = list[0] + 200
        ymin = list[1] + 200
        xmax = list[2] + 200
        ymax = list[3] + 200
        # 四对点的生成
        pcenter = np.array([height / 2, width / 2, 0, 0], np.float32)

        # print(xmin,ymin,xmax,ymax)
        p1 = np.array([xmin, ymin, 0, 0], np.float32) - pcenter
        p2 = np.array([xmin, ymax, 0, 0], np.float32) - pcenter
        p3 = np.array([xmax, ymin, 0, 0], np.float32) - pcenter
        p4 = np.array([xmax, ymax, 0, 0], np.float32) - pcenter
        # 变换后四个点坐标
        # print(p1,p2,p3,p4)
        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)

        list_dst = [dst1, dst2, dst3, dst4]
        # print(dst1,dst2,dst3,dst4)

        dst = np.zeros((4, 2), np.float32)

        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
        # 取四个点min与max作为新的缺陷的标签
        # print(dst)
        xmin = min(dst[0, 0], dst[1, 0], dst[2, 0], dst[3, 0])
        xmax = max(dst[0, 0], dst[1, 0], dst[2, 0], dst[3, 0])
        ymin = min(dst[0, 1], dst[1, 1], dst[2, 1], dst[3, 1])
        ymax = max(dst[0, 1], dst[1, 1], dst[2, 1], dst[3, 1])

        box = [int(xmin), int(ymin), int(xmax), int(ymax)]

        return box

    # 双线性差值算法
    def biLinearInterpolation(self, img, dstH, dstW):
        dstH = int(dstH)
        dstW = int(dstW)
        scrH, scrW, channel = img.shape
        img = np.pad(img, ((0, 1), (0, 1), (0, 0)), 'constant')
        retimg = np.zeros((int(dstH), int(dstW), 3), dtype=np.uint8)
        for i in range(dstH):
            for j in range(dstW):
                scrx = (i + 1) * (scrH / dstH) - 1
                scry = (j + 1) * (scrW / dstW) - 1
                x = math.floor(scrx)
                y = math.floor(scry)
                u = scrx - x
                v = scry - y
                retimg[i, j] = (1 - u) * (1 - v) * img[x, y] + u * (1 - v) * img[x + 1, y] + (1 - u) * v * img[
                    x, y + 1] + u * v * img[x + 1, y + 1]
        return retimg

    # 图像二值化处理
    def binarization(self, img):
        # 变微灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 大津法二值化
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
        dst = cv2.dilate(dst, None, iterations=1)
        # 腐蚀，白区域变小
        dst = cv2.erode(dst, None, iterations=4)

        return dst
    # ----------------------------------------------------------------------------------------
