# coding=utf-8
import os
import time
import cv2
from PIL import Image
from xml.etree import ElementTree as ET

PATH_VOC = '/Users/zijiao/Desktop/Video/VOCdevkit/VOC2012'
PATH_FACE_SAVED = '/Users/zijiao/Desktop/Video/head.txt'
PATH_ANNOTATIONS = os.path.join(PATH_VOC, 'Annotations')
PATH_IMAGE_SETS = os.path.join(PATH_VOC, 'JPEGImages')
# 图片查看间隔
TIME_OFFSET = 5
# 是否加载图片
LOAD_IMAGE = True
# 是否保存带有脸部数据的图片
SAVE_FACE = False


def parse_box(box):
    x1 = int(float(box.findtext('xmin')))
    y1 = int(float(box.findtext('ymin')))
    x2 = int(float(box.findtext('xmax')))
    y2 = int(float(box.findtext('ymax')))
    return x1, y1, x2, y2


if __name__ == '__main__':
    print('Start read image sets.')
    head_path = []
    for xml_file in os.listdir(PATH_ANNOTATIONS)[:]:
        xml_file_path = os.path.join(PATH_ANNOTATIONS, xml_file)
        xml = ET.parse(xml_file_path)

        filename = xml.findtext('./filename')
        image_path = os.path.join(PATH_IMAGE_SETS, filename)
        print(image_path)
        im = None
        if LOAD_IMAGE:
            im = cv2.imread(image_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        show = True

        object = xml.find('./object')
        boxes = object.findall('./bndbox')
        for box in boxes:
            x1, y1, x2, y2 = parse_box(box)
            if LOAD_IMAGE:
                im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

        parts = object.findall('./part')
        for part in parts:
            if part.findtext('name') == 'head':
                head_path.append(xml_file.split('.xml')[0])
                head = part.find('bndbox')
                x1, y1, x2, y2 = parse_box(head)
                if LOAD_IMAGE:
                    im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if LOAD_IMAGE and show:
            im = Image.fromarray(im)
            im.show()
            time.sleep(TIME_OFFSET)
    if SAVE_FACE:
        with open(PATH_FACE_SAVED, 'w') as f:
            for p in head_path:
                f.write(p + '\n')
                f.flush()
                print p
        print len(head_path)
