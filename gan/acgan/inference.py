# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from imageio import imsave

batch_size = 1
z_dim = 128
LABEL = 34

def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m

def get_random_tags():
    y = np.random.uniform(0.0, 1.0, [batch_size, LABEL]).astype(np.float32)
    p_other = [0.6, 0.6, 0.25, 0.04488882, 0.3, 0.05384738]
    for i in range(batch_size):
        for j in range(len(p_other)):
            if y[i, j + 28] < p_other[j]:
                y[i, j + 28] = 1
            else:
                y[i, j + 28] = 0

    phc = [0.15968645, 0.21305391, 0.15491921, 0.10523116, 0.07953927, 0.09508879, 0.03567429, 0.07733163, 0.03157895, 0.01833307, 0.02236442, 0.00537514, 0.00182371]
    phs = [0.52989922,  0.37101264,  0.12567589,  0.00291153,  0.00847864]
    pec = [0.28350664, 0.15760678, 0.17862742, 0.13412254, 0.14212126, 0.0543913, 0.01020637, 0.00617501, 0.03167493, 0.00156775]
    for i in range(batch_size):
        y[i, :28] = 0

        hc = np.random.random()
        for j in range(len(phc)):
            if np.sum(phc[:j]) < hc < np.sum(phc[:j + 1]):
                y[i, j] = 1
                break

        hs = np.random.random()
        for j in range(len(phs)):
            if np.sum(phs[:j]) < hs < np.sum(phs[:j + 1]):
                y[i, j + 13] = 1
                break

        ec = np.random.random()
        for j in range(len(pec)):
            if np.sum(pec[:j]) < ec < np.sum(pec[:j + 1]):
                y[i, j + 18] = 1
                break
    return y

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./anime_acgan-60000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
g = graph.get_tensor_by_name('generator/g/Tanh:0')
noise = graph.get_tensor_by_name('noise:0')
noise_y = graph.get_tensor_by_name('noise_y:0')
is_training = graph.get_tensor_by_name('is_training:0')

# 随机生成样本
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
y_samples = get_random_tags()
gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
gen_imgs = (gen_imgs + 1) / 2
imgs = [img[:, :, :] for img in gen_imgs]
gen_imgs = montage(imgs)
gen_imgs = np.clip(gen_imgs, 0, 1)
imsave('1_二次元头像随机生成.jpg', gen_imgs)

# 生成指定标签的样本
all_tags = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair', 'short hair', 'twintails', 'drill hair', 'ponytail', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes', 'blush', 'smile', 'open mouth', 'hat', 'ribbon', 'glasses']
for i, tags in enumerate([['blonde hair', 'twintails', 'blush', 'smile', 'ribbon', 'red eyes'], ['silver hair', 'long hair', 'blush', 'smile', 'open mouth', 'blue eyes']]):
    z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    y_samples = np.zeros([1, LABEL])
    for tag in tags:
        y_samples[0, all_tags.index(tag)] = 1
    y_samples = np.repeat(y_samples, batch_size, 0)
    gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
    gen_imgs = (gen_imgs + 1) / 2
    imgs = [img[:, :, :] for img in gen_imgs]
    gen_imgs = montage(imgs)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    imsave('%d_二次元头像指定标签.jpg' % (i + 2), gen_imgs)

# 固定噪音随机标签
z_samples = np.random.uniform(-1.0, 1.0, [1, z_dim]).astype(np.float32)
z_samples = np.repeat(z_samples, batch_size, 0)
y_samples = get_random_tags()
gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
gen_imgs = (gen_imgs + 1) / 2
imgs = [img[:, :, :] for img in gen_imgs]
gen_imgs = montage(imgs)
gen_imgs = np.clip(gen_imgs, 0, 1)
imsave('4_二次元头像固定噪音.jpg', gen_imgs)