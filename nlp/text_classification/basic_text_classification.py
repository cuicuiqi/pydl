# ### 基本文本分类
# 官网示例：https://www.tensorflow.org/tutorials/keras/basic_text_classification
# 主要步骤：
#   1.加载IMDB数据集
#   2.探索数据：了解数据格式、将整数转换为字词
#   3.准备数据
#   4.构建模型：隐藏单元、损失函数和优化器
#   5.创建验证集
#   6.训练模型
#   7.评估模型
#   8.可视化：创建准确率和损失随时间变化的图
#
# ### IMDB数据集
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb
# 包含来自互联网电影数据库的50000条影评文本
#

# coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version: {}  - tf.keras version: {}".format(tf.VERSION, tf.keras.__version__))  # 查看版本
ds_path = str(pathlib.Path.cwd()) + "\\datasets\\imdb\\"  # 数据集路径

# ### 查看numpy格式数据
np_data = np.load(ds_path + "imdb.npz")
print("np_data keys: ", list(np_data.keys()))  # 查看所有的键
# print("np_data values: ", list(np_data.values()))  # 查看所有的值
# print("np_data items: ", list(np_data.items()))  # 查看所有的item

# ### 加载IMDB数据集
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    path=ds_path + "imdb.npz",
    num_words=10000  # 保留训练数据中出现频次在前10000位的字词
)

# ### 探索数据：了解数据格式
# 数据集已经过预处理：每个样本都是一个整数数组，表示影评中的字词
# 每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print("First record: {}".format(train_data[0]))  # 第一条影评(影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词)
print("Before len:{} len:{}".format(len(train_data[0]), len(train_data[1])))  # 影评的长度会有所不同
# 将整数转换回字词
word_index = imdb.get_word_index(ds_path + "imdb_word_index.json")  # 整数值与词汇的映射字典
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    """查询包含整数到字符串映射的字典对象"""
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print("The content of first record: ", decode_review(train_data[0]))  # 显示第1条影评的文本

print("The content of first record: ", decode_review(train_data[0]))  # 显示第1条影评的文本

# ### 准备数据
# 影评（整数数组）必须转换为张量，然后才能馈送到神经网络中，而且影评的长度必须相同
# 采用方法：填充数组，使之都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)  # 使用 pad_sequences 函数将长度标准化
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
print("After - len: {} len: {}".format(len(train_data[0]), len(train_data[1])))  # 样本的影评长度都已相同
print("First record: \n", train_data[0])  # 填充后的第1条影评

# ### 构建模型
# 本示例中，输入数据由字词-索引数组构成。要预测的标签是 0 或 1
# 按顺序堆叠各个层以构建分类器（模型有多少层，每个层有多少个隐藏单元）
vocab_size = 10000  # 输入形状（用于影评的词汇数）
model = keras.Sequential()  # 创建一个Sequential模型，然后通过简单地使用.add()方法将各层添加到模型

# Embedding层：在整数编码的词汇表中查找每个字词-索引的嵌入向量
# 模型在接受训练时会学习这些向量，会向输出数组添加一个维度(batch, sequence, embedding)
model.add(keras.layers.Embedding(vocab_size, 16))
# GlobalAveragePooling1D 层通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量
model.add(keras.layers.GlobalAveragePooling1D())
# 长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# 最后一层与单个输出节点密集连接。应用sigmoid激活函数后，结果是介于 0 到 1 之间的浮点值，表示概率或置信水平
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()  # 打印出关于模型的简单描述

# ### 损失函数和优化器
# 模型在训练时需要一个损失函数和一个优化器
# 有多种类型的损失函数，一般来说binary_crossentropy更适合处理概率问题，可测量概率分布之间的“差距”
model.compile(optimizer=tf.train.AdamOptimizer(),  # 优化器
              loss='binary_crossentropy',  # 损失函数
              metrics=['accuracy'])  # 在训练和测试期间的模型评估标准

# ### 创建验证集
# 仅使用训练数据开发和调整模型，然后仅使用一次测试数据评估准确率
# 从原始训练数据中分离出验证集，可用于检查模型处理从未见过的数据的准确率
x_val = train_data[:10000]  # 从原始训练数据中分离出10000个样本，创建一个验证集
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]  # 从原始训练数据中分离出10000个样本，创建一个验证集
partial_y_train = train_labels[10000:]

# ### 训练模型
# 对partial_x_train和partial_y_train张量中的所有样本进行迭代
# 在训练期间，监控模型在验证集(x_val, y_val)的10000个样本上的损失和准确率
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,  # 训练周期（训练模型迭代轮次）
                    batch_size=512,  # 批量大小（每次梯度更新的样本数）
                    validation_data=(x_val, y_val),  # 验证数据
                    verbose=2  # 日志显示模式：0为安静模式, 1为进度条（默认）, 2为每轮一行
                    )  # 返回一个history对象，包含一个字典，其中包括训练期间发生的所有情况

# ### 评估模型
# 在测试模式下返回模型的误差值和评估标准值
results = model.evaluate(test_data, test_labels)  # 返回两个值：损失（表示误差的数字，越低越好）和准确率
print("Result: {}".format(results))

# ### 可视化
history_dict = history.history  # model.fit方法返回一个History回调，它具有包含连续误差的列表和其他度量的history属性
print("Keys: {}".format(history_dict.keys()))  # 4个条目，每个条目对应训练和验证期间的一个受监控指标
loss = history.history['loss']
validation_loss = history.history['val_loss']
accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']
epochs = range(1, len(accuracy) + 1)

plt.subplot(121)  # 创建损失随时间变化的图，作为1行2列图形矩阵中的第1个subplot
plt.plot(epochs, loss, 'bo', label='Training loss')  # 绘制图形， 参数“bo”表示蓝色圆点状（blue dot）
plt.plot(epochs, validation_loss, 'b', label='Validation loss')  # 参数“b”表示蓝色线状（solid blue line）
plt.title('Training and validation loss')  # 标题
plt.xlabel('Epochs')  # x轴标签
plt.ylabel('Loss')  # y轴标签
plt.legend()  # 绘制图例

plt.subplot(122)  # 创建准确率随时间变化的图
plt.plot(epochs, accuracy, color='red', marker='o', label='Training accuracy')
plt.plot(epochs, validation_accuracy, 'r', linewidth=1, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("./outputs/sample-2-figure.png", dpi=200, format='png')
plt.show()  # 显示图形









