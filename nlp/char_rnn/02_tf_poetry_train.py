#https://github.com/chinese-poetry/chinese-poetry


import tensorflow as tf
import numpy as np
import glob
import json
from collections import Counter
from tqdm import tqdm
from snownlp import SnowNLP

poets = []
paths = glob.glob('json/poet.*.json')
for path in paths:
    data = open(path, 'r').read()
    data = json.loads(data)
    for item in data:
        content = ''.join(item['paragraphs'])
        if len(content) >= 24 and len(content) <= 32:
            content = SnowNLP(content)
            poets.append('[' + content.han + ']')

poets.sort(key=lambda x: len(x))
print('共%d首诗' % len(poets), poets[0], poets[-1])


chars = []
for item in poets:
    chars += [c for c in item]
print('共%d个字' % len(chars))

chars = sorted(Counter(chars).items(), key=lambda x:x[1], reverse=True)
print('共%d个不同的字' % len(chars))
print(chars[:10])

chars = [c[0] for c in chars]
char2id = {c: i + 1 for i, c in enumerate(chars)}
id2char = {i + 1: c for i, c in enumerate(chars)}


#整理训练数据
batch_size = 64
X_data = []
Y_data = []

for b in range(len(poets) // batch_size):
    start = b * batch_size
    end = b * batch_size + batch_size
    batch = [[char2id[c] for c in poets[i]] for i in range(start, end)]
    maxlen = max(map(len, batch))
    X_batch = np.full((batch_size, maxlen - 1), 0, np.int32)
    Y_batch = np.full((batch_size, maxlen - 1), 0, np.int32)

    for i in range(batch_size):
        X_batch[i, :len(batch[i]) - 1] = batch[i][:-1]
        Y_batch[i, :len(batch[i]) - 1] = batch[i][1:]

    X_data.append(X_batch)
    Y_data.append(Y_batch)

print(len(X_data), len(Y_data))


#定义模型结构和优化器
hidden_size = 256
num_layer = 2
embedding_size = 256

# tf.compat.v1.disable_eager_execution()
X = tf.placeholder(tf.int32, [batch_size, None])
Y = tf.placeholder(tf.int32, [batch_size, None])
learning_rate = tf.Variable(0.0, trainable=False)

cell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True) for i in range(num_layer)],
    state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

embeddings = tf.Variable(tf.random_uniform([len(char2id) + 1, embedding_size], -1.0, 1.0))
embedded = tf.nn.embedding_lookup(embeddings, X)

# outputs: batch_size, max_time, hidden_size
# last_states: 2 tuple(two LSTM), 2 tuple(c and h)
#              batch_size, hidden_size
outputs, last_states = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)

outputs = tf.reshape(outputs, [-1, hidden_size])                # batch_size * max_time, hidden_size
logits = tf.layers.dense(outputs, units=len(char2id) + 1)       # batch_size * max_time, len(char2id) + 1
logits = tf.reshape(logits, [batch_size, -1, len(char2id) + 1]) # batch_size, max_time, len(char2id) + 1
probs = tf.nn.softmax(logits)                                   # batch_size, max_time, len(char2id) + 1

loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits, Y, tf.ones_like(Y, dtype=tf.float32)))
params = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5)
optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, params))

#训练模型，共训练50轮
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(50):
    sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))

    data_index = np.arange(len(X_data))
    np.random.shuffle(data_index)
    X_data = [X_data[i] for i in data_index]
    Y_data = [Y_data[i] for i in data_index]

    losses = []
    for i in tqdm(range(len(X_data))):
        ls_, _ = sess.run([loss, optimizer], feed_dict={X: X_data[i], Y: Y_data[i]})
        losses.append(ls_)

    print('Epoch %d Loss %.5f' % (epoch, np.mean(losses)))


#保存模型，以便在单机上使用
saver = tf.train.Saver()
saver.save(sess, './poet_generation_tensorflow')

import pickle
with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([char2id, id2char], fw)

#在单机上使用模型生成古诗，可随机生成或生成藏头诗

