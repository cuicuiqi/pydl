"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <TensorFlow项目实战2.x>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""
from functools import partial
import os
import traceback
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

mydataset = __import__("代码8-10  mydataset")
data = mydataset#.data

AttGANmodels = __import__("代码8-11  AttGANmodels")
models = AttGANmodels#.models



img_size = 128
shortcut_layers = 1
inject_layers =1
enc_dim = 64
dec_dim = 64
dis_dim = 64
dis_fc_dim = 1024
enc_layers = 5
dec_layers = 5
dis_layers = 5

batch_size = 32
thres_int = 0.5 #训练时，特征的上下限值域
test_int = 1.0#测试时，特征属性的上下限值域（一般要大于训练时的值域，使特征更加明显）
n_sample = 64
# others
experiment_name = "128_shortcut1_inject1_None"
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
n_att = len(att_default)


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================
import scipy.misc
def imwrite(image, path):
    """Save an [-1.0, 1.0] image."""
    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))



tf.compat.v1.reset_default_graph()

config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#建立数据集
te_data = data.Celeba(r'D:\\01-TF\\01-TF2\\Chapter10\\code10.3\\data\\', att_default, img_size, 1, mode='test', sess=sess)

#构建模型
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)

D = partial(models.D, n_att=n_att, dim=dis_dim, fc_dim=dis_fc_dim, n_layers=dis_layers)


# inputs
xa_sample = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.compat.v1.placeholder(tf.float32, shape=[None, n_att])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================
# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
print(ckpt_dir)
thisckpt_dir = tf.train.latest_checkpoint(ckpt_dir)
print(thisckpt_dir)
restorer = tf.compat.v1.train.Saver()
restorer.restore(sess, thisckpt_dir)

try:

    thisckpt_dir = tf.train.latest_checkpoint(ckpt_dir)
    restorer = tf.compat.v1.train.Saver()
    restorer.restore(sess, thisckpt_dir)
except:
    raise Exception(' [*] No checkpoint!')

  
#######################################################################    
# sample-slide
n_slide  =10  
test_int_min = 0.7
test_int_max = 1.2
test_att = 'Eyeglasses'
try:
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        b_sample_ipt = batch[1]

        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        for i in range(n_slide):
            test_int = (test_int_max - test_int_min) / (n_slide - 1) * i + test_int_min
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            _b_sample_ipt[..., att_default.index(test_att)] = test_int
            x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
        sample = np.concatenate(x_sample_opt_list, 2)

        save_dir = './output/%s/sample_testing_slide_%s' % (experiment_name, test_att)

        os.makedirs(save_dir, exist_ok=True)
        imwrite(sample.squeeze(0), '%s/%d.png' % (save_dir, idx + 182638))
        print('%d.png done!' % (idx + 182638))

except:
    traceback.print_exc()
finally:
    sess.close()    
    
