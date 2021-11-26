import i2v
import cv2
import glob
import os
from imageio import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle

images = glob.glob('characters/*.jpg')
print(len(images))

illust2vec = i2v.make_i2v_with_chainer('model/illust2vec_tag_ver200.caffemodel', 'model/tag_list.json')
cascade = cv2.CascadeClassifier('anime_detect/lbpcascade_animeface.xml')
OUTPUT_DIR = 'faces/'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

num = 0
for x in tqdm(range(len(images))):
    img_path = images[x]
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        x0 = cx - int(0.75 * w)
        x1 = cx + int(0.75 * w)
        y0 = cy - int(0.75 * h)
        y1 = cy + int(0.75 * h)
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= image.shape[1]:
            x1 = image.shape[1] - 1
        if y1 >= image.shape[0]:
            y1 = image.shape[0] - 1
        w = x1 - x0
        h = y1 - y0
        if w > h:
            x0 = x0 + w // 2 - h // 2
            x1 = x1 - w // 2 + h // 2
            w = h
        else:
            y0 = y0 + h // 2 - w // 2
            y1 = y1 - h // 2 + w // 2
            h = w

        face = image[y0: y0 + h, x0: x0 + w, :]
        face = cv2.resize(face, (128, 128))
        cv2.imwrite(os.path.join(OUTPUT_DIR, '%d.jpg' % num), face)
        num += 1
print(num)



fw = open('face_tags.txt', 'w')
tags = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
        'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair',
        'long hair', 'short hair', 'twintails', 'drill hair', 'ponytail',
        'blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes',
        'aqua eyes', 'black eyes', 'orange eyes',
        'blush', 'smile', 'open mouth', 'hat', 'ribbon', 'glasses']
fw.write('id,' + ','.join(tags) + '\n')

images = glob.glob(os.path.join(OUTPUT_DIR, '*.jpg'))
for x in tqdm(range(len(images))):
    img_path = images[x]
    image = imread(img_path)
    result = illust2vec.estimate_specific_tags([image], tags)[0]

    hair_colors = [[h, result[h]] for h in tags[0:13]]
    hair_colors.sort(key=lambda x: x[1], reverse=True)
    for h in tags[0:13]:
        if h == hair_colors[0][0]:
            result[h] = 1
        else:
            result[h] = 0

    hair_styles = [[h, result[h]] for h in tags[13:18]]
    hair_styles.sort(key=lambda x: x[1], reverse=True)
    for h in tags[13:18]:
        if h == hair_styles[0][0]:
            result[h] = 1
        else:
            result[h] = 0

    eye_colors = [[h, result[h]] for h in tags[18:28]]
    eye_colors.sort(key=lambda x: x[1], reverse=True)
    for h in tags[18:28]:
        if h == eye_colors[0][0]:
            result[h] = 1
        else:
            result[h] = 0

    for h in tags[28:]:
        if result[h] > 0.25:
            result[h] = 1
        else:
            result[h] = 0

    fw.write(img_path + ',' + ','.join([str(result[t]) for t in tags]) + '\n')

fw.close()

