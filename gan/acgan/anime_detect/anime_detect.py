# -*- coding: utf-8 -*-

import cv2

cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

image = cv2.imread('imgs/demo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
for i, (x, y, w, h) in enumerate(faces):
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
    cv2.imwrite('face_%d.jpg' % i, face)