from common.utils.conver import *
from common.utils.compute import *
from common.utils.log import *
import random
import cv2
from math import *
import numpy as np


class IndustryFeatureAugment():
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
    def colorTrans(self, defect_img, light=None, saturation=None, tone_r=None, tone_g=None, tone_b=None):
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
        log.info(conver_log("工业缺陷特征模拟色彩变换模型", ["亮度调节参数", "饱和度调节参数", "r通道色调调节参数", "g通道色调调节参数", "b通道色调调节参数"],
                            [light, saturation, tone_r, tone_g, tone_b]))

        # 对缺陷图像进行二值化处理，缺陷位置像素值为0，非缺陷位置像素值为255
        bin_img = self.binarization(defect_img)

        # 获取图像RGB均值与方差
        # RGB_mean[0][1][2]--R均值，G均值，B均值
        # RGB_var[0][1][2]--R方差，G方差，B方差
        RGB_mean, RGB_var = compute_RGB_mean_var(defect_img)
        R_mean, G_mean, B_mean = RGB_mean[0], RGB_mean[1], RGB_mean[2]
        R_var, G_var, B_var = RGB_var[0], RGB_var[1], RGB_var[2]

        # 图像处理前置过程
        rows, cols, channels = defect_img.shape
        img_new = defect_img.copy()

        # 色彩变换公式
        for i in range(rows):
            for j in range(cols):

                # 缺陷部分变换，即二值化后缺陷图像像素点255的部分（缺陷部分）进行色彩变换
                if (bin_img[i, j] == 0):
                    # 获取缺陷位置像素点
                    R = defect_img[i, j][2]
                    B = defect_img[i, j][1]
                    G = defect_img[i, j][0]

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

                # 非缺陷位置不进行变换
                else:
                    img_new[i, j, 0] = defect_img[i, j, 0]
                    img_new[i, j, 1] = defect_img[i, j, 1]
                    img_new[i, j, 2] = defect_img[i, j, 2]

        return img_new

    # ----------------------------------------------------------------------------------------

    # 方位变换模型
    # ----------------------------------------------------------------------------------------
    # 注：
    # 随机位置，主要是标注的变换，用于后续缺陷与原工件图进行融合
    # 参数控制：
    # size：工件图像尺寸；类型：长度为2的list列表；示例：[200,200]
    #       size[0]：工件图像宽度width类型为int，
    #       size[1]：工件图像高度height类型为int；
    # box：缺陷图像位置中心点坐标与尺寸；类型：长度为4的list列表；示例：[0.5,0.5,0.1，0.1]
    #       box[0]：缺陷所处位置中心点x值，x坐标/工件宽度width，double类型
    #       box[1]：缺陷所处位置中心点y值，y坐标/工件高度height，double类型
    #       box[2]：缺陷宽度w，w/工件宽度weight，double类型
    #       box[2]：缺陷宽度h，h/工件宽高度height，double类型
    # 返回值：变换后方位坐标box
    def siteTrans(self, size, box):
        log = Log()

        # 图像处理前置过程
        width = size[0]
        height = size[1]
        new_box = []

        # 将标注转换为四个点坐标形式
        voc_list = coordinateYolo2voc(size, box)
        defect_w = voc_list[2] - voc_list[0]
        defect_h = voc_list[3] - voc_list[1]
        centerx = int((voc_list[2] + voc_list[0]) / 2)
        centery = int((voc_list[3] + voc_list[1]) / 2)

        # 打日志
        log.info(conver_log("工业缺陷特征模拟方位变换模型，变换前",
                            ["工件图像宽度width", "工件图像高度height", "缺陷所处中心点坐标x", "缺陷所处中心点坐标y", "缺陷宽度width", "缺陷高度height"],
                            [width, height, centerx, centery, defect_w, defect_h]))

        # 随机生成缺陷方位的中心点坐标
        x_start = ceil(defect_w / 2)
        x_end = floor(width - defect_w / 2)
        y_start = ceil(defect_h / 2)
        y_end = floor(height - defect_h / 2)

        if (x_start > x_end or y_start > y_end):
            log.info('工业缺陷特征模拟方位变换模型，缺陷图像尺寸大于工件图像尺寸')
            assert ("缺陷图像尺寸大于工件图像尺寸")
        else:
            centerx = random.randint(x_start, x_end)
            centery = random.randint(y_start, y_end)

        # 生成新的坐标点
        xmin = int(centerx - defect_w / 2)
        ymin = int(centery - defect_h / 2)
        xmax = int(centerx + defect_w / 2)
        ymax = int(centery + defect_h / 2)
        new_box.append(xmin)
        new_box.append(ymin)
        new_box.append(xmax)
        new_box.append(ymax)

        log.info(conver_log("工业缺陷特征模拟方位变换模型，变换后",
                            ["工件图像宽度width", "工件图像高度height", "缺陷所处中心点坐标x", "缺陷所处中心点坐标y", "缺陷宽度width", "缺陷高度height"],
                            [width, height, centerx, centery, defect_w, defect_h]))
        # 转化为yolo的box模式
        res = coordinateVoc2yolo(size, new_box)
        # print(res)
        return res

    # ----------------------------------------------------------------------------------------

    # 缩放变换模型
    # ----------------------------------------------------------------------------------------
    # 参数控制：
    # size:工件尺寸，防止因缩放导致缺陷尺寸超出工件尺寸
    # defect_img：缺陷图像；类型：img
    # fx：横向拉伸倍数；取值范围：（0,4），大于1为放大操作，小于1为缩小操作；类型：double；示例：0,5
    # fy：纵向拉伸倍数；取值范围：（0,4），大于1为放大操作，小于1为缩小操作；类型：double；示例：0.5
    # 注：
    # 如果为None值参数则默认在(0.6,1.4)取随机值
    # 返回值：img
    def zoomTrans(self, size, defect_img, fx=None, fy=None):
        log = Log()
        # 参数前置处理
        if (fx is None):
            fx = random.uniform(0.7, 1.3)
        if (fy is None):
            fy = random.uniform(0.7, 1.3)

        # 前置校验
        if (fx <= 0 or fx > 4):
            assert "横向拉伸系数参数范围应在(0,4]"
        if (fy <= 0 or fy > 4):
            assert "纵向拉伸系数参数范围应在(0,4]"

        # 打日志
        log.info(conver_log("工业缺陷特征模拟缩放变换模型", ["横向拉伸系数", "纵向拉伸系数"],
                            [fx, fy]))

        # 图像缩放
        res = cv2.resize(defect_img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

        # 防止缩放后缺陷尺寸超过零件尺寸
        obj_img_width = size[0]
        obj_img_height = size[1]
        res_img_width = res.shape[1]
        res_img_height = res.shape[0]

        if (res_img_width >= obj_img_width):
            fx_new = obj_img_width / res_img_width - 0.07
            log.info(conver_log("工业缺陷特征模拟缩放变换模型，缺陷尺寸大于工件尺寸", ["调整缩小缺陷width尺寸倍数"],
                                [fx_new]))
        else:
            fx_new = 1

        if (res_img_height >= obj_img_height):
            fy_new = obj_img_height / res_img_height - 0.07
            log.info(conver_log("工业缺陷特征模拟缩放变换模型，缺陷尺寸大于工件尺寸", ["调整缩小缺陷height尺寸倍数"],
                                [fy_new]))
        else:
            fy_new = 1

        res = cv2.resize(res, None, fx=fx_new, fy=fy_new, interpolation=cv2.INTER_CUBIC)
        return res

    # ----------------------------------------------------------------------------------------

    # 旋转变换模型
    # ----------------------------------------------------------------------------------------
    # 参数控制：
    # size:工件尺寸，防止因缩放导致缺陷尺寸超出工件尺寸
    # defect_img：缺陷图像；类型：img
    # degree：旋转角度；取值范围：（0,360）；类型：int；示例：90
    # 注：
    # 如果为None值参数则默认在(0,360)取随机值
    # 返回值：img
    def rotateTrans(self, size, defect_img, degree=None):
        log = Log()
        if (degree is None):
            degree = random.randint(0, 360)

        # 前置校验
        if (degree < 0 or degree > 360):
            assert "旋转角度调节系数参数范围应在[0,360]"

        # 打日志
        log.info(conver_log("工业缺陷特征模拟旋转变换模型", ["旋转角度"],
                            [degree]))

        height, width = defect_img.shape[:2]
        # 旋转后的尺寸
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，图像旋转后一部分图像会被接去掉，同样边缘也会产生黑边，把旋转后的图像放置白色背景中，防止丢失像素
        matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

        imgRotation = cv2.warpAffine(defect_img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        res = imgRotation

        # 防止缩放后缺陷尺寸超过零件尺寸
        obj_img_width = size[0]
        obj_img_height = size[1]
        res_img_width = res.shape[1]
        res_img_height = res.shape[0]

        if (res_img_width >= obj_img_width):
            fx_new = obj_img_width / res_img_width - 0.1
            log.info(conver_log("工业缺陷特征模拟旋转变换模型，缺陷width尺寸大于工件尺寸", ["调整缩小缺陷width尺寸倍数"],
                                [fx_new]))
        else:
            fx_new = 1

        if (res_img_height >= obj_img_height):
            fy_new = obj_img_height / res_img_height - 0.1
            log.info(conver_log("工业缺陷特征模拟旋转变换模型，缺陷height尺寸大于工件尺寸", ["调整缩小缺陷height尺寸倍数"],
                                [fy_new]))
        else:
            fy_new = 1

        res = cv2.resize(res, None, fx=fx_new, fy=fy_new, interpolation=cv2.INTER_CUBIC)

        return res

    # ----------------------------------------------------------------------------------------

    # 翻转变换模型
    # ----------------------------------------------------------------------------------------
    # 参数控制：
    # defect_img：缺陷图像；类型：img
    # flip_type：翻转模式；取值范围：[0,1]；类型：int；0为左右翻转，1为上下翻转
    # 注：
    # 如果为None值参数则默认取0或1
    # 返回值：img
    def flipTrans(self, defect_img, flip_type=None):
        log = Log()
        if (flip_type is None):
            flip_type = random.randint(0, 1)
        # 前置校验
        if (flip_type != 0 or flip_type != 1):
            assert "翻转调节参数应该为0或1"

        # 打日志
        log.info(conver_log("工业缺陷特征模拟翻转变换模型", ["翻转调节参数"],
                            [flip_type]))

        if (flip_type == 0):
            img = np.fliplr(defect_img)
            return img
        if (flip_type == 1):
            img = np.flipud(defect_img)
            return img

    # ----------------------------------------------------------------------------------------

    # common_util
    # ----------------------------------------------------------------------------------------
    # 标注变换
    # 主要为旋转变换与缩放变换
    def annotationTrans(self, large_size, small_size, box):
        new_box = anno_compute(large_size, small_size, box)
        x = new_box[0]
        y = new_box[1]
        w = new_box[2]
        h = new_box[3]

        # print(large_size[0]*w,large_size[1]*h)

        # 如果变换后坐标超出原图的尺寸，即按照方位变换重新调整到有效的范围内
        if (x + w / 2 >= 1 or x - w / 2 <= 0 or y + h / 2 >= 1 or y - h / 2 <= 0):
            return self.siteTrans(large_size, new_box)

        return new_box

    # 缺陷擦除
    # 通过缺陷图像box中尺寸以及中心点位置对具有缺陷的工件进行缺陷擦除
    def defectErasure(self, img, box, erasure_type='ns'):
        flags = 0
        if (erasure_type == 'ns'):
            flags = cv2.INPAINT_NS
        if (erasure_type == 'telea'):
            flags = cv2.INPAINT_TELEA

        height, width = img.shape[0:2]
        size = [width, height]
        voc_list = coordinateYolo2voc(size, box)
        xmin = voc_list[0]
        ymin = voc_list[1]
        xmax = voc_list[2]
        ymax = voc_list[3]

        # 生成img大小的全黑色图像
        target = np.zeros((height, width), dtype=np.uint8)

        # 构建mask图像，即要擦除的图像
        # 首先将图像转化RGB图像
        mask = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        for i in range(h):
            for j in range(w):
                m = ymin + i
                n = xmin + j
                # print(m, n)
                mask[m, n, 0] = img[m, n, 0]
                mask[m, n, 1] = img[m, n, 1]
                mask[m, n, 2] = img[m, n, 2]

        # 再将mask图像转化为灰度图
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 擦除
        dst = cv2.inpaint(img, mask, 3, flags)

        return dst

    # 缺陷融合
    # 将缺陷与工件图按照box中的中心位置融合进去
    def defectMerge(self, defect_img, obj, box, merge_type='normal'):
        log = Log()
        # Read images : src image will be cloned into dst
        flags = 2
        if (merge_type == 'normal'):
            flags = cv2.NORMAL_CLONE
        if (merge_type == 'mixed'):
            flags = cv2.MIXED_CLONE
        if (merge_type == 'monochrome'):
            flags = cv2.MONOCHROME_TRANSFER

        # Create an all white mask
        mask = 255 * np.ones(defect_img.shape, defect_img.dtype)

        # The location of the center of the src in the dst
        h, w = obj.shape[0:2]
        centerx = box[0]
        centery = box[1]
        x = int(centerx * w)
        y = int(centery * h)
        # print(h,w,x,y,box[2]*w,box[3]*h)
        # print(defect_img.shape[1])
        # print(defect_img.shape[0])
        center = (x, y)

        # Seamlessly clone src into dst and put the results in output

        mixed = cv2.seamlessClone(defect_img, obj, mask, center, flags)
        return mixed

    # 图像裁剪
    # 通过box来裁剪图像
    def crop(self, img, box):
        height, width = img.shape[0:2]
        size = [width, height]
        voc_list = coordinateYolo2voc(size, box)
        xmin = voc_list[0]
        ymin = voc_list[1]
        xmax = voc_list[2]
        ymax = voc_list[3]
        defect_img = img[ymin:ymax, xmin:xmax]

        return defect_img

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
