import cv2
import os


def read_imgs(img_dir):
    imgs = os.listdir(img_dir)
    return imgs


def read_labels(label_dir):
    txts = os.listdir(label_dir)
    txts = [txt for txt in txts if not txt.split('.')[0] == "classes"]  # 过滤掉classes.txt文件
    return txts


def read_img(img_path, img_name):
    img_file = os.path.join(img_path, img_name)
    img = cv2.imread(img_file, 1)
    return img


def read_label(label_path, label_name):
    # 获取标注文件txt中的标注信息
    all_objects = []
    txt_file = os.path.join(label_path, label_name)
    with open(txt_file, 'r') as f:
        objects = f.readlines()
        for object in objects:
            object = object.strip().split(' ')
            all_objects.append(object)
            # print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']

    return all_objects
