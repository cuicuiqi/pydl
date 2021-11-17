import numpy

print(numpy.__version__)


import scipy

print(scipy.__version__)


import matplotlib

print(matplotlib.__version__)

import PIL

print(PIL.__version__)

import cv2

print(cv2.__version__)

# In[ ]:


import tensorflow

print(tensorflow.__version__)

# In[7]:


# 下载lfw的人脸数据集,并解压
# http://vis-www.cs.umass.edu/lfw/lfw.tgz
import random
from glob import glob
import numpy as np

# 加载所有人脸图像，返回每张图像的路径，形成一个数组，最后将数组通过np.array()转换成NumPy数组
human_filepaths = np.array(glob("lfw/*/*"))
# 对human_filepaths数组变量里的数据通过shuffle()函数打乱混洗
random.shuffle(human_filepaths)

human_filepaths[:10]

print("human_files.shape={}".format(human_filepaths.shape))

import matplotlib.pyplot as plt

# 设置matplotlib在绘图时使用默认样式
plt.style.use('default')
from matplotlib import image

import cv2

# 先使用OpenCV来检测人脸，OpenCV在Github上提供了很多个人脸检测模型，并以XML文件保存在
# 地址：https://github.com/opencv/opencv/tree/master/data/haarcascades
# 这里我们预先下载了一个haarcascade_frontalface_alt.xml的检测模型到本项目的github上。

# 提取OpenCV的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_filepaths[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出人脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的人脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的人脸的识别框
for (x, y, w, h) in faces:
    # 在人脸图像中以矩形形式绘制出识别框
    # 参数1：目标图像
    # 参数2：(x, y)起始坐标
    # 参数3：(x, y)检测到的人脸最大坐标
    # 参数4：绘制框的颜色，因为是BGR顺序，所以第三个255表示red，就是红色的框
    # 参数5：线框的宽度
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()

# In[28]:


# 加载人脸所属类别列表
# glob是一个文件操作相关的模块，通过指定的匹配模式，返回相应的文件或文件夹路径
# 这里的操作就是返回lfw目录下的所有文件夹
# 最后通过列表推导式遍历每个文件路径字符串，并截取人脸所属类别名称那段字符串

from glob import glob

facepath_prefix_len = len('lfw/')
face_names = [item[facepath_prefix_len:] for item in sorted(glob("lfw/*"))]
print("共计有{}个人脸".format(len(face_names)))

# In[29]:


face_names[:10]


# 查看随机9张人脸的图像
# 创建9个绘图对象，3行3列
fig, axes = plt.subplots(nrows=3, ncols=3)
# 设置绘图的总容器大小
fig.set_size_inches(10, 10)

# 随机选择9个数，也就是9张人脸（可能重复，且每次都不一样）
random_9_nums = np.random.choice(len(human_filepaths), 9)
# 从数据集中选出9张图
random_9_imgs = human_filepaths[random_9_nums]
print(random_9_imgs)

# 根据这随机的9张图片路径，截取取得相应的人脸所属人物的名字
imgname_list = []
for imgpath in random_9_imgs:
    imgname = imgpath[facepath_prefix_len:]
    imgname = imgname[:imgname.find('/')]
    imgname_list.append(imgname)

index = 0
for row_index in range(3):  # 行
    for col_index in range(3):  # 列
        # 读取图片的数值内容
        img = image.imread(random_9_imgs[index])
        # 获取绘图Axes对象，根据[行索引, 列索引]
        ax = axes[row_index, col_index]
        # 在Axes对象上显示图像
        ax.imshow(img)
        # 在绘图对象上设置人的名字
        ax.set_xlabel(imgname_list[index])
        # 索引加1
        index += 1


# 对数据集进行遍历，读取每张图片，并获取它的大小，对比每张图片是否大小一致
faces_shape_list = []
for filepath in human_filepaths:
    shape = image.imread(filepath).shape
    if len(shape) == 3 and shape[0] == 250 and shape[1] == 250 and shape[2] == 3:
        faces_shape_list.append(shape)
    else:
        print("找到一张异样大小的人脸图片。路径是：{}".format(filepath))

faces_shapes = np.asarray(faces_shape_list)

print("总共{}张。".format(len(faces_shapes)))
print("随机抽取三张图片的维度是{}。".format(faces_shapes[np.random.choice(len(faces_shapes), 3)]))
