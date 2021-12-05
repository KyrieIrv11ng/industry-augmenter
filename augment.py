import configparser
from common.utils.input import *
from model.industryFeatureAugment import *
from model.industryImagingAugment import *
from common.utils.configEnum import *


class Augment():
    def augment_model(self):
        log = Log()
        # 读取配置文件
        configpath = os.path.join((os.path.dirname(os.path.realpath(__file__))), 'config')
        configfile = os.path.join(configpath, 'config.ini')
        # print(configpath, configfile)
        cf = configparser.ConfigParser()

        cf.read(configfile, encoding='UTF-8')  # 读取配置文件，如果写文件的绝对路径，就可以不用os模块

        img_input_dir = cf.get("InputAndOutput", "img_input_dir")
        anno_input_dir = cf.get("InputAndOutput", "anno_imput_dir")
        img_output_dir = cf.get("InputAndOutput", "img_output_dir")
        anno_output_dir = cf.get("InputAndOutput", "anno_output_dir")

        if not os.path.exists(img_output_dir):
            os.mkdir(img_output_dir)
        if not os.path.exists(anno_output_dir):
            os.mkdir(anno_output_dir)

        # 配置文件中读取总控制开关初始值
        iia_switch = cf.get("TotalControl", "IIAswitch")
        ifa_switch = cf.get("TotalControl", "IFAswitch")

        # 把上面的两个循环改写成为一个循环：
        imgs = os.listdir(img_input_dir)
        txts = os.listdir(anno_input_dir)
        txts = [txt for txt in txts if not txt.split('.')[0] == "classes"]  # 过滤掉classes.txt文件
        # 注意，这里保持图片的数量和标签txt文件数量相等，且要保证名字是一一对应的   (后面改进，通过判断txt文件名是否在imgs中即可)
        if len(imgs) == len(txts):  # 注意：./Annotation_txt 不要把classes.txt文件放进去
            map_imgs_txts = [(img, txt) for img, txt in zip(imgs, txts)]
            img_index = 0
            for img_name, txt_name in map_imgs_txts:

                img_index = img_index + 1
                objects = read_label(anno_input_dir, txt_name)
                img = read_img(img_input_dir, img_name)
                log.info(conver_log('开始对第' + str(img_index) + '张图像进行扩增', ['图像名称', '标注名称'], [img_name, txt_name]))

                # iia总控制开关
                if (iia_switch == ConfigEnum.true.value):
                    img_new = img.copy()
                    objects_new = objects

                    log.info(conver_log('通过工业图像采集过程模拟方法进行扩增', ['图像名称', '标注名称'], [img_name, txt_name]))

                    iia = IndustryImagingAugment()
                    # iia模块控制开关初始化，读取配置文件
                    color_switch = cf.get("IndustryImagingAugment-colorTrans", "switch")
                    perspective_switch = cf.get("IndustryImagingAugment-perspectiveTrans", "switch")
                    resolution_switch = cf.get("IndustryImagingAugment-resolutionTrans", "switch")

                    if (color_switch == ConfigEnum.true.value):
                        # 获取参数
                        light = cf.get("IndustryImagingAugment-colorTrans", "light")
                        saturation = cf.get("IndustryImagingAugment-colorTrans", "saturation")
                        tone_r = cf.get("IndustryImagingAugment-colorTrans", "tone_r")
                        tone_g = cf.get("IndustryImagingAugment-colorTrans", "tone_g")
                        tone_b = cf.get("IndustryImagingAugment-colorTrans", "tone_b")

                        img_new = iia.colorTrans(img_new,
                                                 None if light == ConfigEnum.random.value else float(light),
                                                 None if saturation == ConfigEnum.random.value else float(saturation),
                                                 None if tone_r == ConfigEnum.random.value else float(tone_r),
                                                 None if tone_g == ConfigEnum.random.value else float(tone_g),
                                                 None if tone_b == ConfigEnum.random.value else float(tone_b))
                        objects_new = objects
                    if (resolution_switch == ConfigEnum.true.value):
                        # 获取参数
                        s = cf.get("IndustryImagingAugment-resolutionTrans", "s")

                        img_new = iia.resolutionTrans(img_new,
                                                      None if s == ConfigEnum.random.value else float(s))
                        objects_new = objects

                    if (perspective_switch == ConfigEnum.true.value):
                        # 获取参数
                        anglex = cf.get("IndustryImagingAugment-perspectiveTrans", "anglex")
                        angley = cf.get("IndustryImagingAugment-perspectiveTrans", "angley")
                        anglez = cf.get("IndustryImagingAugment-perspectiveTrans", "anglez")
                        H = cf.get("IndustryImagingAugment-perspectiveTrans", "H")

                        img_new, objects_new = iia.perspectiveTrans(img_new,
                                                                    objects,
                                                                    None if anglex == ConfigEnum.random.value else int(
                                                                        anglex),
                                                                    None if angley == ConfigEnum.random.value else int(
                                                                        angley),
                                                                    None if anglez == ConfigEnum.random.value else int(
                                                                        anglez),
                                                                    None if H == ConfigEnum.random.value else float(H)
                                                                    )
                    name = img_name.split('.')[0]
                    name_new = name + '_iia'
                    img_new_name = name_new + '.jpg'
                    txt_new_name = name_new + '.txt'
                    img_new_path = os.path.join(img_output_dir, img_new_name)
                    txt_new_path = os.path.join(anno_output_dir, txt_new_name)

                    # 保存扩增图像
                    cv2.imwrite(img_new_path, img_new)

                    # 保存扩增标注
                    out_txt_f = open(txt_new_path, 'w')
                    for object in objects_new:
                        out_txt_f.write(str(object[0]) + " " + " ".join(
                            [str(object[1]), str(object[2]), str(object[3]), str(object[4])]) + '\n')

                # ifa总控制开关
                if (ifa_switch == ConfigEnum.true.value):
                    img_new = img.copy()
                    objects_new = []
                    ifa = IndustryFeatureAugment()
                    log.info(conver_log('通过工业缺陷特征模拟方法进行扩增', ['图像名称', '标注名称'], [img_name, txt_name]))

                    # iia模块控制开关初始化，读取配置文件
                    color_switch = cf.get("IndustryFeatureAugment-colorTrans", "switch")
                    zoom_switch = cf.get("IndustryFeatureAugment-zoomTrans", "switch")
                    rotate_switch = cf.get("IndustryFeatureAugment-rotateTrans", "switch")
                    flip_switch = cf.get("IndustryFeatureAugment-flipTrans", "switch")
                    site_switch = cf.get("IndustryFeatureAugment-siteTrans", "switch")

                    if (zoom_switch == ConfigEnum.true.value or rotate_switch == ConfigEnum.true.value):
                        anno_trans_flag = True
                    else:
                        anno_trans_flag = False

                    height, width = img.shape[0:2]
                    size = [width, height]
                    # 先进行缺陷擦除
                    for object in objects:
                        box = []
                        box.append(object[1])
                        box.append(object[2])
                        box.append(object[3])
                        box.append(object[4])

                        # 裁剪缺陷
                        img_defect = ifa.crop(img, box)

                        erasure_type = cf.get("ErasureAndMergeType", "erasure_type")
                        img_new = ifa.defectErasure(img_new, box, erasure_type)
                        # print(img_new.shape[1],img_new.shape[0])
                    # 新的objects

                    # 再对缺陷进行变换处理
                    for object in objects:
                        box = []
                        box.append(float(object[1]))
                        box.append(float(object[2]))
                        box.append(float(object[3]))
                        box.append(float(object[4]))

                        # 裁剪缺陷
                        img_defect = ifa.crop(img, box)

                        if (color_switch == ConfigEnum.true.value):
                            light = cf.get("IndustryFeatureAugment-colorTrans", "light")
                            saturation = cf.get("IndustryFeatureAugment-colorTrans", "saturation")
                            tone_r = cf.get("IndustryFeatureAugment-colorTrans", "tone_r")
                            tone_g = cf.get("IndustryFeatureAugment-colorTrans", "tone_g")
                            tone_b = cf.get("IndustryFeatureAugment-colorTrans", "tone_b")

                            img_defect = ifa.colorTrans(img_defect,
                                                        None if light == ConfigEnum.random.value else float(light),
                                                        None if saturation == ConfigEnum.random.value else float(
                                                            saturation),
                                                        None if tone_r == ConfigEnum.random.value else float(tone_r),
                                                        None if tone_g == ConfigEnum.random.value else float(tone_g),
                                                        None if tone_b == ConfigEnum.random.value else float(tone_b)
                                                        )
                        if (zoom_switch == ConfigEnum.true.value):
                            fx = cf.get("IndustryFeatureAugment-zoomTrans", "fx")
                            fy = cf.get("IndustryFeatureAugment-zoomTrans", "fy")
                            img_defect = ifa.zoomTrans(size,
                                                       img_defect,
                                                       None if fx == ConfigEnum.random.value else float(fx),
                                                       None if fy == ConfigEnum.random.value else float(fy)
                                                       )

                        if (rotate_switch == ConfigEnum.true.value):
                            degree = cf.get("IndustryFeatureAugment-rotateTrans", "degree")
                            img_defect = ifa.rotateTrans(size,
                                                         img_defect,
                                                         None if degree == ConfigEnum.random.value else int(degree)
                                                         )

                        if (flip_switch == ConfigEnum.true.value):
                            flip_type = cf.get("IndustryFeatureAugment-flipTrans", "flip_type")
                            img_defect = ifa.flipTrans(img_defect,
                                                       None if flip_type == ConfigEnum.random.value else int(flip_type))

                        if (anno_trans_flag):
                            large_size = size
                            small_size = [img_defect.shape[1], img_defect.shape[0]]
                            # print(small_size)
                            box_new = ifa.annotationTrans(large_size, small_size, box)

                        else:
                            box_new = box

                        if (site_switch == ConfigEnum.true.value):
                            box_new = ifa.siteTrans(size, box_new)

                        object_new = []
                        object_new.append(object[0])
                        object_new.append(box_new[0])
                        object_new.append(box_new[1])
                        object_new.append(box_new[2])
                        object_new.append(box_new[3])

                        objects_new.append(object_new)
                        # print(img_defect.shape[1],img_defect.shape[0])
                        # print(img_new.shape[1],img_new.shape[0])
                        # print(box)
                        # print(box_new)
                        # cv2.imshow("1",img_new)
                        # cv2.waitKey(0)
                        # cv2.imshow("2",img_defect)
                        # cv2.waitKey(0)
                        merge_type = cf.get("ErasureAndMergeType", "merge_type")

                        img_new = ifa.defectMerge(img_defect, img_new, box_new, merge_type)

                    name = img_name.split('.')[0]
                    name_new = name + '_ifa'
                    img_new_name = name_new + '.jpg'
                    txt_new_name = name_new + '.txt'
                    img_new_path = os.path.join(img_output_dir, img_new_name)
                    txt_new_path = os.path.join(anno_output_dir, txt_new_name)

                    # 保存扩增图像
                    cv2.imwrite(img_new_path, img_new)

                    # 保存扩增标注
                    out_txt_f = open(txt_new_path, 'w')
                    for object in objects_new:
                        out_txt_f.write(str(object[0]) + " " + " ".join(
                            [str(object[1]), str(object[2]), str(object[3]), str(object[4])]) + '\n')
