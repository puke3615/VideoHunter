"""
图片预处理
"""

import os
import cv2
import imutils
import numpy as np
import urllib.request as request
from multiprocessing import Pool


def parse_people_names(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError('File "%s" not found.' % file_path)
    with open(file_path) as f:
        people_names = []
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            if line:
                people_names.append(line)
        index2name = {i: n for i, n in enumerate(people_names)}
        name2index = {v: k for k, v in index2name.items()}
        return people_names, index2name, name2index


def parse_images(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError('File "%s" not found.' % file_path)
    with open(file_path) as f:
        headers = []
        images_data = []
        for row, line in enumerate(f):
            if row == 0:
                continue
            line = line.strip()
            if row == 1:
                headers = line.split('\t')[1:]
            else:
                images_data.append(line.split('\t'))
        return headers, images_data


def download(url, path):
    try:
        parent = os.path.dirname(path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        # print(url, path)
        request.urlretrieve(url, path)
        print('下载成功 %s' % url)
    except Exception as e:
        print('下载失败 %s' % url)


def download_images(images_data, name2index, download_dir, process_num=5):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    pool = Pool(process_num)
    for data in images_data:
        type = '%03d' % name2index[data[0]]
        id = data[1]
        url = data[2]
        rect = data[3]
        rect_str = '_'.join(rect.split(','))
        postfix = url.split('.')[-1]

        file_name = '%04d_%s.%s' % (int(id), rect_str, postfix)
        save_path = os.path.join(download_dir, type, file_name)
        # print(save_path)

        pool.apply_async(download, (url, save_path))
    pool.close()
    pool.join()


def show(image, show_box=True, width=None):
    if not os.path.exists(image):
        raise FileNotFoundError('File "%s" not found.' % image)
    im = cv2.imread(image)
    basename = os.path.basename(image)
    scale = 1.
    if width is not None:
        scale = width / float(im.shape[1])
        im = imutils.resize(im, width)
    if show_box:
        rect = [int(int(v) * scale) for v in basename.split('.')[0].split('_')[1:]]
        start = tuple(rect[:2])
        end = tuple(rect[2:])
        cv2.rectangle(im, start, end, (0, 0, 255), 2)
    cv2.imshow(basename, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clip_face(from_dir, to_dir):
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    for subdir in os.listdir(from_dir):
        subdir_path = os.path.join(from_dir, subdir)
        for file in os.listdir(subdir_path):
            try:
                origin_file_path = os.path.join(subdir_path, file)
                prefix, postfix = file.split('.')
                id = prefix.split('_')[0]
                face_file_name = '%s.%s' % (id, postfix)
                face_dir = os.path.join(to_dir, subdir)
                if not os.path.exists(face_dir):
                    os.makedirs(face_dir)
                face_file_path = os.path.join(face_dir, face_file_name)
                x1, y1, x2, y2 = [int(v) for v in prefix.split('_')[1:]]
                im = cv2.imread(os.path.join(subdir_path, file))
                depth = im.shape[-1]
                face = im[y1: y2 + 1, x1: x2 + 1, :]
                cv2.imwrite(face_file_path, face)
                print('裁剪成功：%s' % origin_file_path)
            except Exception as e:
                print('裁剪失败：%s' % origin_file_path)


PATH_PEOPLE = 'dev_people.txt'
PATH_URLS = 'dev_urls.txt'
PATH_DOWNLOAD = 'images'
PATH_FACE = 'face'

if __name__ == '__main__':
    # 读人名
    # names, index2name, name2index = parse_people_names(PATH_PEOPLE)
    # print(names)

    # 读取图片信息
    # headers, images = parse_images(PATH_URLS)
    # print(headers)
    # print(len(images))
    # print(images[0])

    # 下载图片
    # download_images(images, name2index, PATH_DOWNLOAD, process_num=20)

    # 显示图片
    path = 'images/001/0044_205_131_379_305.jpg'
    show(path, show_box=True, width=None)

    # 裁剪脸部区域
    # clip_face(PATH_DOWNLOAD, PATH_FACE)

