#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy 
print(numpy.__version__)


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[ ]:


import matplotlib
print(matplotlib.__version__)


# In[ ]:


import tensorflow 
print(tensorflow.__version__)


# In[ ]:


import keras 
print(keras.__version__)


# In[ ]:


import tqdm 
print(tqdm.__version__)


# In[ ]:


import PIL 
print(PIL.__version__)


# In[ ]:


from sklearn import datasets

filepath = "dataset"

data = datasets.load_files(filepath)
filename_list = data["filenames"]
target_list = data["target"]
target_name_list = data["target_names"]


# In[ ]:


dir(data)


# In[ ]:


filename_list.shape


# In[ ]:


filename_list[:20]


# In[ ]:


target_list


# In[ ]:


target_name_list


# # 随机9张病理图像预览

# In[ ]:


import matplotlib.pyplot as plt
# 设置matplotlib在绘图时的默认样式
plt.style.use('default')

from matplotlib import image


# In[ ]:


import numpy as np

# 创建9个绘图对象，3行3列
fig, axes = plt.subplots(nrows=3, ncols=3)
# 设置绘图的总容器大小
fig.set_size_inches(10, 9)

# 随机选择9个数，也就是9张病理图像（可能重复，且每次都不一样）
random_9_nums = np.random.choice(len(filename_list), 9)
# 从数据集中选出9张图和它的路径
random_9_imgs = filename_list[random_9_nums]
print(random_9_imgs)

# 根据这随机的9张图片路径，截取取得相应的皮肤癌病理名称
imgname_list = []
for imgpath in random_9_imgs:
    imgname = imgpath[len(filepath) + 1:] 
    imgname = imgname[:imgname.find('/')]
    imgname_list.append(imgname)

index = 0
for row_index in range(3): # 行
    for col_index in range(3): # 列
        # 读取图片的数值内容
        img = image.imread(random_9_imgs[index])
        # 获取绘图Axes对象，根据[行索引, 列索引]
        ax = axes[row_index, col_index]
        # 在Axes对象上显示图像
        ax.imshow(img)
        # 在绘的图下面设置显示皮肤癌病理名称
        ax.set_xlabel(imgname_list[index])
        # 索引加1
        index += 1


# In[ ]:


def print_img_shape(i, filepath):
    shape = image.imread(filepath).shape
    print("第{}张的shape是{}".format(i + 1, shape))
    
print("查看病理图像的大小：\r")
for i, img_path in enumerate(random_9_imgs):
    print_img_shape(i, img_path)


# In[ ]:


from sklearn import model_selection

# 分割训练数据集和测试数据集
X_train, X_test, y_train, y_test = model_selection.train_test_split(filename_list, target_list, test_size=0.2)

# 将测试集数据分割一半给验证集
half_test_count = int(len(X_test) / 2)
X_valid = X_test[:half_test_count]
y_valid = y_test[:half_test_count]

X_test = X_test[half_test_count:]
y_test = y_test[half_test_count:]

print("X_train.shape={}, y_train.shape={}.".format(X_train.shape, y_train.shape))
print("X_valid.shape={}, y_valid.shape={}.".format(X_valid.shape, y_valid.shape))
print("X_test.shape={}, y_test.shape={}.".format(X_test.shape, y_test.shape))


# In[ ]:


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    """
    定义一个函数，将每张图片都转换成卷积神经网络期待的大小(1, 224, 224, 3)
    """
    # 加载图片使用PIL库的load_img()方法，它返回一个PIL对象
    img = image.load_img(img_path, target_size=(224, 224, 3))
    # 将PIL图片对象类型转化为格式(224, 224, 3)的3维数组
    x = image.img_to_array(img)
    # 将3维数组转化格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    """
    定义一个函数，将数组里的所有路径的图片都转换成图像数值类型并返回
    """
    # tqdm模块表示使用进度条显示，传入一个所有图片的数组对象
    # 将所有图片的对象一个个都转换成numpy数值对象张量后，并返回成数组
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    # 将对象垂直堆砌排序摆放
    return np.vstack(list_of_tensors)


# In[ ]:


import numpy as np
from PIL import ImageFile 
# 为了防止PIL读取图片对象时出现IO错误，则设置截断图片为True
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# 将所有图片都转换成标准大小的数值图像对象，然后除以255，
# 进行归一化处理（简单说，就是将颜色值转换成0到1之间的值，便于神经网络计算和处理）
# RGB的颜色值，最大为255，最小为0
# 对训练集数据进行处理
train_tensors = paths_to_tensor(X_train).astype(np.float32) / 255
# 对验证集数据进行处理
valid_tensors = paths_to_tensor(X_valid).astype(np.float32) / 255
# 对测试集数据进行处理
test_tensors = paths_to_tensor(X_test).astype(np.float32) / 255


# In[ ]:


from keras import backend as K 
K.clear_session()


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

# 图像的shape
input_shape = train_tensors[0].shape
# 有多少个类别
num_classes = len(target_name_list)

# 创建Sequential模型
model = Sequential()

# 创建输入层，输入层必须传入input_shape参数以表示图像大小
# 这里输入深度从16开始，padding填充使用same（也就是不够kernel大小的使用0填充）
# 滑动窗口使用1x1的大小，窗口移动跨步大小也是1x1的
model.add(Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                 activation='relu', input_shape=input_shape))
# 添加最大池化层，卷积层大小一致是1x1，有效填充范围默认是valid
model.add(MaxPooling2D(pool_size=(1, 1)))
# 添加Dropout层，每次丢弃50%的网络节点，防止过拟合
model.add(Dropout(0.5))

# 添加卷积层，深度是32，内核大小是1x1，跨步是1x1，使用relu来调节神经网络
model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.3))

# 添加卷积层，深度是64
model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))

# 添加卷积层，深度是128
model.add(Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))

# 添加全局平均池化层，对空间数据进行处理
model.add(GlobalAveragePooling2D())
# 添加Dropout，每次丢弃50%
model.add(Dropout(0.5))
# 添加输出层，有3个类别输出
model.add(Dense(num_classes, activation="softmax"))
                 
# 打印输出网络模型架构
model.summary()


# In[ ]:


# 编译模型
# Loss Function的sparse_categorical_crossentropy和categorical_crossentropy的区别是
# 如果targets的值经过了one-hot encoding处理，那么损失函数就用categorical_crossentropy
# 如果targets的值是数值，未经过one-hot encoding处理，那么损失函数就用sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 比如我们这里使用sparse_categorical_crossentropy作为损失函数，所以，以下的one-hot encoding就不需要运行
# from keras import utils
# y_train = utils.to_categorical(y_train)
# y_valid = utils.to_categorical(y_valid)
# y_test = utils.to_categorical(y_test)


# In[ ]:


get_ipython().system('mkdir saved_models')


# In[ ]:


from keras.callbacks import ModelCheckpoint 

# 创建检查点对象
checkpointer = ModelCheckpoint(filepath='saved_models/skin_cancer.best_weights.hdf5', 
                               verbose=1, 
                               save_best_only=True)
epochs = 10
model.fit(train_tensors, 
          y_train, 
          validation_data=(valid_tensors, y_valid),
          epochs=epochs, 
          batch_size=20, 
          callbacks=[checkpointer], 
          verbose=1)


# In[ ]:


# 加载刚才训练的权重到模型中
model.load_weights("saved_models/skin_cancer.best_weights.hdf5") 
# 评估模型精确度
score = model.evaluate(test_tensors, y_test, verbose=1)
print("Test {}: {:.2f}. Test {}: {:.2f}.".format(model.metrics_names[0], 
                                                 score[0]*100, 
                                                 model.metrics_names[1], 
                                                 score[1]*100))


# In[ ]:


# 离开以上环境时，可以单独运行这块代码进行图像分类
from keras.models import load_model
# 加载模型
model = load_model('saved_models/skin_cancer.best_weights.hdf5')

# 加载一张病理图像来测试模型精确度
test_img_path = "nevus_ISIC_0007332.jpg"
# 将图像转换成4维的NumPy数值数组
image_tensor = path_to_tensor(test_img_path)
# 归一化 转换成0到1之间的数值
image_tensor = image_tensor.astype(np.float32) / 255
# 模型预测概率
predicted_result = model.predict(image_tensor)
# 打印输出概率
print(predicted_result)


# In[ ]:


import matplotlib

def draw_predicted_figure(img_path, X, y):
    """
    绘制测试图像和显示预测概率
    """
    # 创建一个绘图对象
    fig, ax = plt.subplots()
    # 设置绘图的总容器大小
    fig.set_size_inches(5, 5)
    # 拼接病理图像对应的名称和它的概率值的字符串
    fig_title = "\n".join(["{}: {:.2f}%\n".format(n, y[i]) for i, n in enumerate(X)])
    # 设置在图像右上角的注解文字
    ax.text(1.01, 0.7, 
            fig_title, 
            horizontalalignment='left', 
            verticalalignment='bottom',
            transform=ax.transAxes)
    # 读取图片的数值内容
    img = matplotlib.image.imread(img_path)
    # 在Axes对象上显示图像
    ax.imshow(img)


# In[ ]:


draw_predicted_figure(test_img_path, target_name_list, predicted_result[0])


# In[ ]:





# In[ ]:





# In[1]:


get_ipython().system('git clone https://github.com/21-projects-for-deep-learning/Simple_Transfer_Learning.git')


# In[ ]:


get_ipython().system('python -m retrain   --bottleneck_dir=tf_files/bottlenecks   --how_many_training_steps=1000   --learning_rate=0.05   --model_dir=tf_files/models/   --summaries_dir=tf_files/training_summaries/"inception_v3"   --output_graph=tf_files/retrained_graph.pb   --output_labels=tf_files/retrained_labels.txt   --architecture="inception_v3"   --image_dir=dataset_small ')


# In[ ]:


get_ipython().system('python -m label_image   --graph=tf_files/retrained_graph.pb    --image=nevus_ISIC_0007332.jpg')

