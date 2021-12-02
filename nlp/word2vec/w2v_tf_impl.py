# -*- coding: utf-8 -*-

import pickle
import numpy as np
import tensorflow as tf
import collections
from tqdm import tqdm

with open('wiki.zh.word.text', 'rb') as fr:
    lines = fr.readlines()
print('共%d行' % len(lines))
print(lines[0].decode('utf-8'))

lines = [line.decode('utf-8') for line in lines]
words = ' '.join(lines)
words = words.replace('\n', '').split(' ')
print('共%d个词' % len(words))

vocab_size = 50000
vocab = collections.Counter(words).most_common(vocab_size - 1)

count = [['UNK', 0]]
count.extend(vocab)
print(count[:10])

word2id = {}
id2word = {}
for i, w in enumerate(count):
    word2id[w[0]] = i
    id2word[i] = w[0]
print(id2word[100], word2id['数学'])

data = []
for i in tqdm(range(len(lines))):
    line = lines[i].strip('\n').split(' ')
    d = []
    for word in line:
        if word in word2id:
            d.append(word2id[word])
        else:
            d.append(0)
            count[0][1] += 1
    data.append(d)
print('UNK数量%d' % count[0][1])

#准备训练数据
X_train = []
Y_train = []
window = 3
for i in tqdm(range(len(data))):
    d = data[i]
    for j in range(len(d)):
        start = j - window
        end = j + window
        if start < 0:
            start = 0
        if end >= len(d):
            end = len(d) - 1

        while start <= end:
            if start == j:
                start += 1
                continue
            else:
                X_train.append(d[j])
                Y_train.append(d[start])
                start += 1
X_train = np.squeeze(np.array(X_train))
Y_train = np.squeeze(np.array(Y_train))
Y_train = np.expand_dims(Y_train, -1)
print(X_train.shape, Y_train.shape)

#定义模型参数
batch_size = 128
embedding_size = 128
valid_size = 16
valid_range = 100
valid_examples = np.random.choice(valid_range, valid_size, replace=False)
num_negative_samples = 64

#定义模型
X = tf.placeholder(tf.int32, shape=[batch_size], name='X')
Y = tf.placeholder(tf.int32, shape=[batch_size, 1], name='Y')
valid = tf.placeholder(tf.int32, shape=[None], name='valid')

embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, X)

nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=Y, inputs=embed, num_sampled=num_negative_samples, num_classes=vocab_size))

optimizer = tf.train.AdamOptimizer().minimize(loss)

#将词向量归一化，并计算和给定词之间的相似度
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

#训练模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())

offset = 0
losses = []
for i in tqdm(range(1000000)):
    if offset + batch_size >= X_train.shape[0]:
        offset = (offset + batch_size) % X_train.shape[0]

    X_batch = X_train[offset: offset + batch_size]
    Y_batch = Y_train[offset: offset + batch_size]

    _, loss_ = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
    losses.append(loss_)

    if i % 2000 == 0 and i > 0:
        print('Iteration %d Average Loss %f' % (i, np.mean(losses)))
        losses = []

    if i % 10000 == 0:
        sim = sess.run(similarity, feed_dict={valid: valid_examples})
        for j in range(valid_size):
            valid_word = id2word[valid_examples[j]]
            top_k = 5
            nearests = (-sim[j, :]).argsort()[1: top_k + 1]
            s = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                s += ' ' + id2word[nearests[k]]
            print(s)

    offset += batch_size

#保存模型、最终词向量、映射字典
saver = tf.train.Saver()
saver.save(sess, './tf_128')

final_embeddings = sess.run(normalized_embeddings)
with open('tf_128.pkl', 'wb') as fw:
    pickle.dump({'embeddings': final_embeddings, 'word2id': word2id, 'id2word': id2word}, fw, protocol=4)



#加载库和得到的词向量、映射字典
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

with open('tf_128.pkl', 'rb') as fr:
    data = pickle.load(fr)
    final_embeddings = data['embeddings']
    word2id = data['word2id']
    id2word = data['id2word']

#获取频次最高的前200个非单字词，对其词向量进行tSNE降维可视化
word_indexs = []
count = 0
plot_only = 200
for i in range(1, len(id2word)):
    if len(id2word[i]) > 1:
        word_indexs.append(i)
        count += 1
        if count == plot_only:
            break

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[word_indexs, :])
labels = [id2word[i] for i in word_indexs]

plt.figure(figsize=(15, 12))
for i, label in enumerate(labels):
    x, y = two_d_embeddings[i, :]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), ha='center', va='top', fontproperties='Microsoft YaHei')
plt.savefig('词向量降维可视化.png')
