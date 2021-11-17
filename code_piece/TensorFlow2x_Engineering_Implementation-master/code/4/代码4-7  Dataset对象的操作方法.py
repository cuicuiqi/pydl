"""
@author: ����ҽ�������� 
@���ںţ�xiangyuejiqiren   �����и����������¼�ѧϰ���ϣ�
@��Դ: <TensorFlow��Ŀʵս2.x>���״��� 
@���״��뼼��֧�֣�bbs.aianaconda.com  
"""



# dataset ops
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import *
import numpy as np


###############  range(*args)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.range(5)
iterator = dataset.make_one_shot_iterator()			#�ӵ���β��һ��
one_element = iterator.get_next()					#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:	# �����Ự��session��

    for i in range(5):		#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element))				#����sess.run����Tensorֵ
'''



###############  zip(datasets)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = Dataset.zip((dataset1,dataset2))
iterator = dataset.make_one_shot_iterator()			#�ӵ���β��һ��
one_element = iterator.get_next()					#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:	# �����Ự��session��

    for i in range(5):		#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element))				#����sess.run����Tensorֵ
'''



###############  concatenate(dataset)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = dataset1.concatenate(dataset2)
iterator = dataset.make_one_shot_iterator()			#�ӵ���β��һ��
one_element = iterator.get_next()					#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:	# �����Ự��session��

    for i in range(10):		#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element))				#����sess.run����Tensorֵ
'''



###############  repeat(count=None)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.repeat(2)
iterator = dataset.make_one_shot_iterator()			#�ӵ���β��һ��
one_element = iterator.get_next()					#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:	# �����Ự��session��

    for i in range(10):		#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element))				#����sess.run����Tensorֵ
'''


###############  shuffle(buffer_size,seed=None,reshuffle_each_iteration=None)


'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.shuffle(1000)
iterator = dataset.make_one_shot_iterator()			#�ӵ���β��һ��
one_element = iterator.get_next()					#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:	# �����Ự��session��

    for i in range(5):		#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element))				#����sess.run����Tensorֵ

'''




###############  batch(count=None)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.batch(batch_size=2)
iterator = dataset.make_one_shot_iterator()			#�ӵ���β��һ��
one_element = iterator.get_next()					#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:	# �����Ự��session��
	while True:
	    for i in range(2):		#ͨ��forѭ����ӡ���е�����
	        print(sess.run(one_element))				#����sess.run����Tensorֵ
'''

###############  padded_batch

'''
data1 = tf.data.Dataset.from_tensor_slices([[1, 2],[1,3]])
data1 = data1.padded_batch(2,padded_shapes=[4])
iterator = data1.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess2:
    print(sess2.run(init_op))
    print("batched data 1:",sess2.run(next_element))
'''

###############  flat_map(map_func)




'''
import numpy as np

##���ڴ�����������
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = tf.data.Dataset.from_tensor_slices(np.array([[1,2,3],[4,5,6]]))

dataset = dataset.flat_map(lambda x: Dataset.from_tensors(x)) 			
iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(10):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element))			#����sess.run����Tensorֵ
'''



######interleave(map_func,cycle_length,block_length=1)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.interleave(lambda x: Dataset.from_tensors(x).repeat(3),
             cycle_length=2, block_length=2)			
iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(100):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element),end=' ')			#����sess.run����Tensorֵ
'''

######filter(predicate)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.filter(lambda x: tf.less(x, 3))			
iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(100):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element),end=' ')			#����sess.run����Tensorֵ

#���˵�ȫΪ0��Ԫ��
dataset = tf.data.Dataset.from_tensor_slices([ [0, 0],[ 3.0, 4.0] ])
dataset = dataset.filter(lambda x: tf.greater(tf.reduce_sum(x), 0))		  #���˵�ȫΪ0��Ԫ��	
iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(100):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element),end=' ')			#����sess.run����Tensorֵ

#���˵������ַ���(1)����һ���ж���
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha��" ])

def _parse_data(line):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [line], tf.bool)
    #tf.cast(isokstr,tf.bool)[0]

    return line,isokstr#tf.cast(isokstr,tf.bool)[0]
dataset = dataset.map(_parse_data)

dataset = dataset.filter(lambda x,y: y)		  #���˵�ȫΪ0��Ԫ��	
iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(100):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element),end=' ')			#����sess.run����Tensorֵ

#���˵������ַ���(2)��ʵ��
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha��" ])

def myfilter(x):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [x], tf.bool)
    return isokstr
dataset = dataset.filter(myfilter)		  #���˵�ȫΪ0��Ԫ��	
#dataset = dataset.filter(lambda x,y: y)		  #���˵�ȫΪ0��Ԫ��	
iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(100):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element),end=' ')			#����sess.run����Tensorֵ

'''
######apply(transformation_func)
'''
data1 = np.arange(50).astype(np.int64)
dataset = tf.data.Dataset.from_tensor_slices(data1)
#�����ݼ���ż�����������зֿ�����window_sizeΪ���ڴ�С��һ��ȡwindow_size��ż���к�window_size�������С���window_size�У���batchΪ���ν��зָ
dataset = dataset.apply((tf.contrib.data.group_by_window(key_func=lambda x: x%2, reduce_func=lambda _, els: els.batch(10), window_size=20)  ))

iterator = dataset.make_one_shot_iterator()		#�ӵ���β��һ��
one_element = iterator.get_next()				#��iterator��ȡ��һ��Ԫ��
with tf.Session() as sess:						#�����Ự��session��
    for i in range(100):							#ͨ��forѭ����ӡ���е�����
        print(sess.run(one_element),end=' ')			#����sess.run����Tensorֵ
'''





