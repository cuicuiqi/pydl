"""
@author: ����ҽ�������� 
@���ںţ�xiangyuejiqiren   �����и����������¼�ѧϰ���ϣ�
@��Դ: <TensorFlow��Ŀʵս2.x>���״��� 
@���״��뼼��֧�֣�bbs.aianaconda.com  
"""

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle
tf.compat.v1.disable_v2_behavior()

def load_sample(sample_dir):
    '''�ݹ��ȡ�ļ���ֻ֧��һ���������ļ�������ֵ��ǩ����ֵ��Ӧ�ı�ǩ��'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#�ݹ�����ļ���
        for filename in filenames:                            #���������ļ���
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #����ļ���
            labelsnames.append( dirpath.split('\\')[-1] )#����ļ�����Ӧ�ı�ǩ

    lab= list(sorted(set(labelsnames)))  #���ɱ�ǩ�����б�
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #�����ֵ�

    labels = [labdict[i] for i in labelsnames]
    return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)


data_dir = 'mnist_digits_images\\'  #�����ļ�·��

(image,label),labelsnames = load_sample(data_dir)   #�����ļ��������ǩ
print(len(image),image[:2],len(label),label[:2])#���load_sample���ص����ݽ��
print(labelsnames[ label[:2] ],labelsnames)#���load_sample���صı�ǩ�ַ���


def get_batches(image,label,input_w,input_h,channels,batch_size):

    queue = tf.compat.v1.train.slice_input_producer([image,label])  #ʹ��tf.train.slice_input_producerʵ��һ������Ķ���
    label = queue[1]                                        #������������ȡ��ǩ

    image_c = tf.io.read_file(queue[0])                        #������������ȡimage·��

    image = tf.image.decode_bmp(image_c,channels)           #����·����ȡͼƬ

    image = tf.image.resize_with_crop_or_pad(image,input_w,input_h) #�޸�ͼƬ��С


    image = tf.image.per_image_standardization(image) #ͼ���׼������(x - mean) / adjusted_stddev

    image_batch,label_batch = tf.compat.v1.train.batch([image,label],#����tf.train.batch����������������
               batch_size = batch_size,
               num_threads = 64)

    images_batch = tf.cast(image_batch,tf.float32)   #����������ת��Ϊfloat32

    labels_batch = tf.reshape(label_batch,[batch_size])#�޸ı�ǩ����״shape
    return images_batch,labels_batch


batch_size = 16
image_batches,label_batches = get_batches(image,label,28,28,1,batch_size)



def showresult(subplot,title,thisimg):          #��ʾ����ͼƬ
    p =plt.subplot(subplot)
    p.axis('off')
    #p.imshow(np.asarray(thisimg[0], dtype='uint8'))
    p.imshow(np.reshape(thisimg, (28, 28)))
    p.set_title(title)

def showimg(index,label,img,ntop):   #��ʾ
    plt.figure(figsize=(20,10))     #������ʾͼƬ�Ŀ���
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)  #��ʼ��

    coord = tf.train.Coordinator()          #�����ж�
    threads = tf.compat.v1.train.start_queue_runners(sess = sess,coord = coord)
    try:
        for step in np.arange(10):
            if coord.should_stop():
                break
            images,label = sess.run([image_batches,label_batches]) #ע������

            showimg(step,label,images,batch_size)       #��ʾͼƬ
            print(label)                                 #��ӡ����

    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()

    coord.join(threads)                             #�ر��ж�

