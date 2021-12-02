# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char] = pickle.load(fr)

batch_size = 1
hidden_size = 256
num_layer = 2
embedding_size = 256

X = tf.placeholder(tf.int32, [batch_size, None])
Y = tf.placeholder(tf.int32, [batch_size, None])
learning_rate = tf.Variable(0.0, trainable=False)

cell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True) for i in range(num_layer)],
    state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

embeddings = tf.Variable(tf.random_uniform([len(char2id) + 1, embedding_size], -1.0, 1.0))
embedded = tf.nn.embedding_lookup(embeddings, X)

outputs, last_states = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)

outputs = tf.reshape(outputs, [-1, hidden_size])
logits = tf.layers.dense(outputs, units=len(char2id) + 1)
probs = tf.nn.softmax(logits)
targets = tf.reshape(Y, [-1])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
params = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5)
optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, params))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./'))


def generate():
    states_ = sess.run(initial_state)

    gen = ''
    c = '['
    while c != ']':
        gen += c
        x = np.zeros((batch_size, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        c = id2char[pos]

    return gen[1:]


def generate_with_head(head):
    states_ = sess.run(initial_state)

    gen = ''
    c = '['
    i = 0
    while c != ']':
        gen += c
        x = np.zeros((batch_size, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))

        if (c == '[' or c == '。' or c == '，') and i < len(head):
            c = head[i]
            i += 1
        else:
            c = id2char[pos]

    return gen[1:]


print(generate())
print(generate_with_head('深度学习'))
