"""
@author: ����ҽ�������� 
@���ںţ�xiangyuejiqiren   �����и����������¼�ѧϰ���ϣ�
@��Դ: <TensorFlow��Ŀʵս2.x>���״��� 
@���״��뼼��֧�֣�bbs.aianaconda.com  
"""

import os
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

tf.compat.v1.disable_v2_behavior()

def load_sample(sample_dir,shuffleflag = True):
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
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)



directory='man_woman\\'                                                     #��������·��
(filenames,labels),_ = load_sample(directory,shuffleflag=False)   #�����ļ��������ǩ


def makeTFRec(filenames,labels): #���庯������TFRecord
    writer= tf.io.TFRecordWriter("mydata.tfrecords") #ͨ��tf.io.TFRecordWriter д�뵽TFRecords�ļ�
    for i in tqdm( range(0,len(labels) ) ):
        img=Image.open(filenames[i])
        img = img.resize((256, 256))
        img_raw=img.tobytes()#��ͼƬת��Ϊ�����Ƹ�ʽ
        example = tf.train.Example(features=tf.train.Features(feature={
                #���ͼƬ�ı�ǩlabel
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                #��ž����ͼƬ
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example�����label��image���ݽ��з�װ

        writer.write(example.SerializeToString())  #���л�Ϊ�ַ���
    writer.close()  #���ݼ��������

makeTFRec(filenames,labels)

################��tf���ݼ�ת��ΪͼƬ##########################
def read_and_decode(filenames,flag = 'train',batch_size = 3):
    #�����ļ�������һ������
    if flag == 'train':
        filename_queue = tf.compat.v1.train.string_input_producer(filenames)#Ĭ���Ѿ���shuffle����ѭ����ȡ
    else:
        filename_queue = tf.compat.v1.train.string_input_producer(filenames,num_epochs = 1,shuffle = False)

    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #�����ļ������ļ�
    features = tf.io.parse_single_example(serialized=serialized_example, #ȡ������image��label��feature����
                                       features={
                                           'label': tf.io.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.io.FixedLenFeature([], tf.string),
                                       })

    #tf.decode_raw���Խ��ַ���������ͼ���Ӧ����������
    image = tf.io.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [256,256,3])
    #
    label = tf.cast(features['label'], tf.int32)

    if flag == 'train':
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5     #��һ��
        img_batch, label_batch = tf.compat.v1.train.batch([image, label],   #������ʹ��tf.train.shuffle_batch������������
                                                batch_size=batch_size, capacity=20)
#        img_batch, label_batch = tf.train.shuffle_batch([image, label],
#                                        batch_size=batch_size, capacity=20,
#                                        min_after_dequeue=10)
        return img_batch, label_batch

    return image, label

#############################################################
TFRecordfilenames = ["mydata.tfrecords"]
image, label =read_and_decode(TFRecordfilenames,flag='test')  #�Բ��Եķ�ʽ�����ݼ�


saveimgpath = 'show\\'    #���屣��ͼƬ·��
if tf.io.gfile.exists(saveimgpath):  #�������saveimgpath������ɾ��
    tf.io.gfile.rmtree(saveimgpath)  #Ҳ����ʹ��shutil.rmtree(saveimgpath)
tf.io.gfile.makedirs(saveimgpath)    #����saveimgpath·��

#��ʼһ���Ự��ȡ����
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())   #��ʼ�����ر�����û�����ᱨ��
    #�������߳�
    coord=tf.train.Coordinator()
    threads= tf.compat.v1.train.start_queue_runners(coord=coord)
    myset = set([])

    try:
        i = 0
        while True:
            example, examplelab = sess.run([image,label])#�ڻỰ��ȡ��image��label
            examplelab = str(examplelab)
            if examplelab not in myset:
                myset.add(examplelab)
                tf.io.gfile.makedirs(saveimgpath+examplelab)
                print(saveimgpath+examplelab,i)
            img=Image.fromarray(example, 'RGB')#ת��Image��ʽ
            img.save(saveimgpath+examplelab+'/'+str(i)+'_Label_'+'.jpg')#����ͼƬ
            print( i)
            i = i+1
    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")
#############################################################
#ѵ����ʽ
image, label =read_and_decode(TFRecordfilenames)  #��ѵ���ķ�ʽ�����ݼ�
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.local_variables_initializer())   #��ʼ�����ر�����û�����ᱨ��
    #�������߳�
    coord=tf.train.Coordinator()
    threads= tf.compat.v1.train.start_queue_runners(coord=coord)
    myset = set([])
    try:
        for i in range(5):
            example, examplelab = sess.run([image,label])#�ڻỰ��ȡ��image��label

            dirtrain = saveimgpath+"train_"+str(i)
            print(dirtrain,examplelab)
            tf.io.gfile.makedirs(dirtrain)
            for lab in range(len(examplelab)):
                print(lab)
                img=Image.fromarray(example[lab], 'RGB')#����Image��֮ǰ�ᵽ��
                img.save(dirtrain+'/'+str(lab)+'_Label_'+str(examplelab[lab])+'.jpg')#����ͼƬ

    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")