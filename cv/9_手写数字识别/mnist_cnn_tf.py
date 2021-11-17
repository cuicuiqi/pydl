import tensorflow
print(tensorflow.__version__)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载并加载数据集
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 为了便于读取，我们把数据集先各自使用一个变量指向它们
x_train, y_train = mnist.train.images, mnist.train.labels
x_valid, y_valid = mnist.validation.images, mnist.validation.labels
x_test, y_test = mnist.test.images, mnist.test.labels

print("训练集图像大小：{}".format(x_train.shape))
print("训练集标签大小：{}".format(y_train.shape))
print("验证集图像大小：{}".format(x_valid.shape))
print("验证集标签大小：{}".format(y_valid.shape))
print("测试集图像大小：{}".format(x_test.shape))
print("测试集标签大小：{}".format(y_test.shape))

# 参数准备
img_size = 28 * 28
num_classes = 10
learning_rate = 1e-4
epochs = 10
batch_size = 50

# 定义输入占位符
x = tf.placeholder(tf.float32, shape=[None, img_size])
x_shaped = tf.reshape(x, [-1, 28, 28, 1])

# 定义输出占位符
y = tf.placeholder(tf.float32, shape=[None, num_classes])

# 定义卷积函数
def create_conv2d(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # 卷积的过滤器大小结构是[filter_height, filter_width, in_channels, out_channels]
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    
    # 定义权重Tensor变量，初始化时是截断正态分布，标准差是0.03
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name+"_W")
    
    # 定义偏移项Tensor变量，初始化时是截断正态分布
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+"_b")
    
    # 定义卷积层
    out_layer = tf.nn.conv2d(input_data, weights, (1, 1, 1, 1), padding="SAME")
    out_layer += bias
    # 通过激活函数ReLU来计算输出
    out_layer = tf.nn.relu(out_layer)
    # 添加最大池化层
    out_layer = tf.nn.max_pool(out_layer, ksize=(1, pool_shape[0], pool_shape[1], 1), strides=(1, 2, 2, 1), padding="SAME")
    return out_layer

# 添加第一层卷积层
layer1 = create_conv2d(x_shaped, 1, 32, (5, 5), (2, 2), name="layer1")
# 添加第二层卷积层
layer2 = create_conv2d(layer1, 32, 64, (5, 5), (2, 2), name="layer2")
# 添加扁平化层
flattened = tf.reshape(layer2, (-1, 7 * 7 * 64))

# 添加全连接层
wd1 = tf.Variable(tf.truncated_normal((7 * 7 * 64, 1000), stddev=0.03), name="wd1")
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name="bd1")
dense_layer1 = tf.add(tf.matmul(flattened, wd1), bd1)
dense_layer1 = tf.nn.relu(dense_layer1)

# 添加输出全连接层
wd2 = tf.Variable(tf.truncated_normal((1000, num_classes), stddev=0.03), name="wd2")
bd2 = tf.Variable(tf.truncated_normal([num_classes], stddev=0.01), name="bd2")
dense_layer2 = tf.add(tf.matmul(dense_layer1, wd2), bd2)

# 添加激活函数的softmax输出层
y_ = tf.nn.softmax(dense_layer2)

# 通过softmax交叉熵定义计算损失值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
# 定义优化器是Adam
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 定义预测结果的比较
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 定义预测的精确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

iteration = 0


import math

# 定义要保存训练模型的变量
saver = tf.train.Saver()

# 创建TensorFlow会话
with tf.Session() as sess:
  
    # 初始化TensorFlow的全局变量
    sess.run(tf.global_variables_initializer())
    
    # 计算所有的训练集需要被训练多少次，当每批次是batch_size个时
    batch_count = int(math.ceil(x_train.shape[0] / float(batch_size)))
    
    # 要迭代epochs次训练
    for e in range(epochs):
        # 对每张图像进行训练
        for batch_i in range(batch_count):
            # 每次取出batch_size张图像
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            # 训练模型
            _, loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            
            # 每训练20次图像时打印一次日志信息，也就是20次乘以batch_size个图像已经被训练了
            if batch_i % 20 == 0:
                print("Epoch: {}/{}".format(e+1, epochs), 
                      "Iteration: {}".format(iteration), 
                      "Training loss: {:.5f}".format(loss))
            iteration += 1
            
            # 每迭代一次时，做一次验证，并打印日志信息
            if iteration % batch_size == 0:
                valid_acc = sess.run(accuracy, feed_dict={x: x_valid, y: y_valid})
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Validation Accuracy: {:.5f}".format(valid_acc))

    # 保存模型的检查点
    saver.save(sess, "checkpoints/mnist_cnn_tf.ckpt")

1e-4

# 预测测试数据集
saver = tf.train.Saver()
with tf.Session() as sess:
    # 从TensorFlow会话中恢复之前保存的模型检查点
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
    
    # 通过测试集预测精确度
    test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
    print("test accuracy: {:.5f}".format(test_acc))


