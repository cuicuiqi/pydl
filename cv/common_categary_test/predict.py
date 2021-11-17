# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

model_dir = '/tmp'
model_name = 'output_graph.pb'
image_dir = './test'
label_dir = '/tmp'
label_filename = 'output_labels.txt'


# 读取并创建一个图graph来存放Google训练好的Inception_v3模型（函数）
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')


# 读取标签labels
def load_labels(label_file_dir):
    if not tf.gfile.Exists(label_file_dir):
        # 预先检测地址是否存在
        tf.logging.fatal('File does not exist %s', label_file_dir)
    else:
        # 读取所有的标签返并回一个list
        labels = tf.gfile.GFile(label_file_dir).readlines()
        for i in range(len(labels)):
            labels[i] = labels[i].strip('\n')
    return labels


# 创建graph
create_graph()

# 创建会话，因为是从已有的Inception_v3模型中恢复，所以无需初始化
with tf.Session() as sess:

    # for n in sess.graph_def.node:
    #     print(n.name)

    # Inception_v3模型的最后一层final_result:0的输出
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # 遍历目录
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            # 输入图像（jpg格式）数据，得到softmax概率值（一个shape=(1,1008)的向量）
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            # 将结果转为1维数据
            predictions = np.squeeze(predictions)

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            print(predictions)
            label_index = predictions.argmax()
            label_name = load_labels(os.path.join(
                label_dir, label_filename))[label_index]
            label_score = predictions[label_index]
            print('%s (score = %.5f)' % (label_name, label_score))
