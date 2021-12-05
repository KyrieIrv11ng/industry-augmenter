import unittest
from model.industryFeatureAugment import *
from common.utils.compute import *
from common.utils.input import *
from model.industryImagingAugment import *


class TestCase(unittest.TestCase):

    # 色彩变换模型测试用例
    def testColorTrans(self):
        ifa = IndustryFeatureAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)
        width, height = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = ifa.crop(img, box)

            img_new_new = ifa.colorTrans(img_new)
            cv2.imshow("1", img_new_new)
            cv2.waitKey(0)

    # 方位变换模型测试用例
    def testSiteTrans(self):
        ifa = IndustryFeatureAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)
        width, height = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            new_box = ifa.siteTrans(size, box)
            print(new_box)

    # 缩放变换模型测试用例
    def testZoomTrans(self):
        ifa = IndustryFeatureAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)
        height, width = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = ifa.crop(img, box)

            img_new_new = ifa.zoomTrans(size, img_new)
            cv2.imshow("1", img_new_new)
            cv2.waitKey(0)

    # 旋转变换模型测试用例
    def testRotateTrans(self):
        ifa = IndustryFeatureAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)
        height, width = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = ifa.crop(img, box)

            img_new_new = ifa.rotateTrans(size, img_new)
            cv2.imshow("1", img_new_new)
            cv2.waitKey(0)

    # 翻转变换模型测试用例
    def testFlipTrans(self):
        ifa = IndustryFeatureAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)
        width, height = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = ifa.crop(img, box)

            img_new_new = ifa.flipTrans(img_new)
            cv2.imshow("1", img_new_new)
            cv2.waitKey(0)

    # 擦除与融合模型测试用例
    def testDefectEerasureAndMerge(self):
        ifa = IndustryFeatureAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'crazing_1.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'crazing_1.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)

        copy_img = img.copy()
        height, width = img.shape[0:2]
        size = [width, height]
        # 先消除原图中所有缺陷
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            copy_img = ifa.defectErasure(copy_img, box)
            cv2.imshow("1", copy_img)
            cv2.waitKey(0)
        # 再将缺陷重新融合
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = ifa.crop(img, box)

            box_new = ifa.siteTrans(size, box)

            copy_img = ifa.defectMerge(img_new, copy_img, box_new)
            print(copy_img.shape[1], copy_img.shape[0])
            print(img_new.shape[1], img_new.shape[0])
            print(box)
            print(box_new)
            cv2.imshow("1", copy_img)
            cv2.waitKey(0)
            cv2.imshow("2", img_new)
            cv2.waitKey(0)

            cv2.imshow("1", copy_img)
            cv2.waitKey(0)

    # 视角变换模型测试用例
    def testPerspectiveTrans(self):
        iia = IndustryImagingAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)

        width, height = img.shape[0:2]
        size = [width, height]

        img_new, objects_new = iia.perspectiveTrans(img, objects, 0, 0, 45)
        for object in objects_new:
            box = []
            box.append(float(object[1]))
            box.append(float(object[2]))
            box.append(float(object[3]))
            box.append(float(object[4]))

            box_2 = coordinateYolo2voc([img_new.shape[1], img_new.shape[0]], box)

            img_new[box_2[1], box_2[0], 0] = 155
            img_new[box_2[1], box_2[0], 1] = 232
            img_new[box_2[1], box_2[0], 2] = 18

            img_new[box_2[3], box_2[2], 0] = 155
            img_new[box_2[3], box_2[2], 1] = 232
            img_new[box_2[3], box_2[2], 2] = 18
            cv2.imshow("1", img_new)
            cv2.waitKey(0)
        print(objects)

    # 色彩变换模型测试用例
    def testColorTrans2(self):
        iia = IndustryImagingAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)

        width, height = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = iia.colorTrans(img)

            cv2.imshow("1", img_new)
            cv2.waitKey(0)

    # 分辨率变换模型测试用例
    def testResolutionTrans(self):
        iia = IndustryImagingAugment()
        # 测试图像与标注
        img_path = r'F:\industy_augmenter\test\images'
        img_name = 'patches_239.jpg'
        anno_path = r'F:\industy_augmenter\test\annotations'
        anno_name = 'patches_239.txt'
        objects = read_label(anno_path, anno_name)
        img = read_img(img_path, img_name)

        width, height = img.shape[0:2]
        size = [width, height]
        for object in objects:
            box = []
            box.append(object[1])
            box.append(object[2])
            box.append(object[3])
            box.append(object[4])

            # 测试缺陷裁剪方法
            img_new = iia.resolutionTrans(img, None)

            cv2.imshow("1", img_new)
            cv2.waitKey(0)
