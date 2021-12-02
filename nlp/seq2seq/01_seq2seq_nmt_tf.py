import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
import os
from tqdm import tqdm
import pickle

def load_vocab(path):
    with open(path, 'r') as fr:
        vocab = fr.readlines()
        vocab = [w.strip('\n') for w in vocab]
    return vocab

vocab_ch = load_vocab('data/vocab.ch')
vocab_en = load_vocab('data/vocab.en')
print(len(vocab_ch), vocab_ch[:20])
print(len(vocab_en), vocab_en[:20])

word2id_ch = {w: i for i, w in enumerate(vocab_ch)}
id2word_ch = {i: w for i, w in enumerate(vocab_ch)}
word2id_en = {w: i for i, w in enumerate(vocab_en)}
id2word_en = {i: w for i, w in enumerate(vocab_en)}


def load_data(path, word2id):
    with open(path, 'r') as fr:
        lines = fr.readlines()
        sentences = [line.strip('\n').split(' ') for line in lines]
        sentences = [[word2id['<s>']] + [word2id[w] for w in sentence] + [word2id['</s>']]
                     for sentence in sentences]

        lens = [len(sentence) for sentence in sentences]
        maxlen = np.max(lens)
        return sentences, lens, maxlen


# train: training, no beam search, calculate loss
# eval: no training, no beam search, calculate loss
# infer: no training, beam search, calculate bleu
mode = 'train'

train_ch, len_train_ch, maxlen_train_ch = load_data('data/train.ch', word2id_ch)
train_en, len_train_en, maxlen_train_en = load_data('data/train.en', word2id_en)
dev_ch, len_dev_ch, maxlen_dev_ch = load_data('data/dev.ch', word2id_ch)
dev_en, len_dev_en, maxlen_dev_en = load_data('data/dev.en', word2id_en)
test_ch, len_test_ch, maxlen_test_ch = load_data('data/test.ch', word2id_ch)
test_en, len_test_en, maxlen_test_en = load_data('data/test.en', word2id_en)

maxlen_ch = np.max([maxlen_train_ch, maxlen_dev_ch, maxlen_test_ch])
maxlen_en = np.max([maxlen_train_en, maxlen_dev_en, maxlen_test_en])
print(maxlen_ch, maxlen_en)

if mode == 'train':
    train_ch = pad_sequences(train_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    train_en = pad_sequences(train_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print(train_ch.shape, train_en.shape)
elif mode == 'eval':
    dev_ch = pad_sequences(dev_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    dev_en = pad_sequences(dev_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print(dev_ch.shape, dev_en.shape)
elif mode == 'infer':
    test_ch = pad_sequences(test_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    test_en = pad_sequences(test_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print(test_ch.shape, test_en.shape)

#定义四个placeholder，对输入进行嵌入
X = tf.placeholder(tf.int32, [None, maxlen_ch])
X_len = tf.placeholder(tf.int32, [None])
Y = tf.placeholder(tf.int32, [None, maxlen_en])
Y_len = tf.placeholder(tf.int32, [None])
Y_in = Y[:, :-1]
Y_out = Y[:, 1:]

k_initializer = tf.contrib.layers.xavier_initializer()
e_initializer = tf.random_uniform_initializer(-1.0, 1.0)

embedding_size = 512
hidden_size = 512

if mode == 'train':
    batch_size = 128
else:
    batch_size = 16

with tf.variable_scope('embedding_X'):
    embeddings_X = tf.get_variable('weights_X', [len(word2id_ch), embedding_size], initializer=e_initializer)
    embedded_X = tf.nn.embedding_lookup(embeddings_X, X)  # batch_size, seq_len, embedding_size

with tf.variable_scope('embedding_Y'):
    embeddings_Y = tf.get_variable('weights_Y', [len(word2id_en), embedding_size], initializer=e_initializer)
    embedded_Y = tf.nn.embedding_lookup(embeddings_Y, Y_in)  # batch_size, seq_len, embedding_size


#定义encoder部分，使用双向LSTM
def single_cell(mode=mode):
    if mode == 'train':
        keep_prob = 0.8
    else:
        keep_prob = 1.0
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell

def multi_cells(num_layers):
    cells = []
    for i in range(num_layers):
        cell = single_cell()
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


with tf.variable_scope('encoder'):
    num_layers = 1
    fw_cell = multi_cells(num_layers)
    bw_cell = multi_cells(num_layers)
    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_X, dtype=tf.float32,
                                                           sequence_length=X_len)
    # fw: batch_size, seq_len, hidden_size
    # bw: batch_size, seq_len, hidden_size
    print('=' * 100, '\n', bi_outputs)

    encoder_outputs = tf.concat(bi_outputs, -1)
    print('=' * 100, '\n', encoder_outputs)  # batch_size, seq_len, 2 * hidden_size

    # 2 tuple(fw & bw), 2 tuple(c & h), batch_size, hidden_size
    print('=' * 100, '\n', bi_state)

    encoder_state = []
    for i in range(num_layers):
        encoder_state.append(bi_state[0][i])  # forward
        encoder_state.append(bi_state[1][i])  # backward
    encoder_state = tuple(encoder_state)  # 2 tuple, 2 tuple(c & h), batch_size, hidden_size
    print('=' * 100)
    for i in range(len(encoder_state)):
        print(i, encoder_state[i])

#定义decoder部分，使用两层LSTM
with tf.variable_scope('decoder'):
    beam_width = 10
    memory = encoder_outputs

    if mode == 'infer':
        memory = tf.contrib.seq2seq.tile_batch(memory, beam_width)
        X_len = tf.contrib.seq2seq.tile_batch(X_len, beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
        bs = batch_size * beam_width
    else:
        bs = batch_size

    attention = tf.contrib.seq2seq.LuongAttention(hidden_size, memory, X_len, scale=True)  # multiplicative
    # attention = tf.contrib.seq2seq.BahdanauAttention(hidden_size, memory, X_len, normalize=True) # additive
    cell = multi_cells(num_layers * 2)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention, hidden_size, name='attention')
    decoder_initial_state = cell.zero_state(bs, tf.float32).clone(cell_state=encoder_state)

    with tf.variable_scope('projected'):
        output_layer = tf.layers.Dense(len(word2id_en), use_bias=False, kernel_initializer=k_initializer)

    if mode == 'infer':
        start = tf.fill([batch_size], word2id_en['<s>'])
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, embeddings_Y, start, word2id_en['</s>'],
                                                       decoder_initial_state, beam_width, output_layer)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                            output_time_major=True,
                                                                            maximum_iterations=2 * tf.reduce_max(X_len))
        sample_id = outputs.predicted_ids
    else:
        helper = tf.contrib.seq2seq.TrainingHelper(embedded_Y, [maxlen_en - 1 for b in range(batch_size)])
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                            output_time_major=True)
        logits = outputs.rnn_output
        logits = tf.transpose(logits, (1, 0, 2))
        print(logits)


if mode != 'infer':
    with tf.variable_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_out, logits=logits)
        mask = tf.sequence_mask(Y_len, tf.shape(Y_out)[1], tf.float32)
        loss = tf.reduce_sum(loss * mask) / batch_size

if mode == 'train':
    learning_rate = tf.Variable(0.0, trainable=False)
    params = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads, params))


#训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if mode == 'train':
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_diy'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(OUTPUT_DIR)

    epochs = 20
    for e in range(epochs):
        total_loss = 0
        total_count = 0

        start_decay = int(epochs * 2 / 3)
        if e <= start_decay:
            lr = 1.0
        else:
            decay = 0.5 ** (int(4 * (e - start_decay) / (epochs - start_decay)))
            lr = 1.0 * decay
        sess.run(tf.assign(learning_rate, lr))

        train_ch, len_train_ch, train_en, len_train_en = shuffle(train_ch, len_train_ch, train_en, len_train_en)

        for i in tqdm(range(train_ch.shape[0] // batch_size)):
            X_batch = train_ch[i * batch_size: i * batch_size + batch_size]
            X_len_batch = len_train_ch[i * batch_size: i * batch_size + batch_size]
            Y_batch = train_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = len_train_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = [l - 1 for l in Y_len_batch]

            feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
            _, ls_ = sess.run([optimizer, loss], feed_dict=feed_dict)

            total_loss += ls_ * batch_size
            total_count += np.sum(Y_len_batch)

            if i > 0 and i % 100 == 0:
                writer.add_summary(sess.run(summary,
                                            feed_dict=feed_dict),
                                   e * train_ch.shape[0] // batch_size + i)
                writer.flush()

        print('Epoch %d lr %.3f perplexity %.2f' % (e, lr, np.exp(total_loss / total_count)))
        saver.save(sess, os.path.join(OUTPUT_DIR, 'nmt'))

if mode == 'eval':
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_diy'
    saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))

    total_loss = 0
    total_count = 0
    for i in tqdm(range(dev_ch.shape[0] // batch_size)):
        X_batch = dev_ch[i * batch_size: i * batch_size + batch_size]
        X_len_batch = len_dev_ch[i * batch_size: i * batch_size + batch_size]
        Y_batch = dev_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = len_dev_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = [l - 1 for l in Y_len_batch]

        feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
        ls_ = sess.run(loss, feed_dict=feed_dict)

        total_loss += ls_ * batch_size
        total_count += np.sum(Y_len_batch)

    print('Dev perplexity %.2f' % np.exp(total_loss / total_count))

if mode == 'infer':
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_diy'
    saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))


    def translate(ids):
        words = [id2word_en[i] for i in ids]
        if words[0] == '<s>':
            words = words[1:]
        if '</s>' in words:
            words = words[:words.index('</s>')]
        return ' '.join(words)


    fw = open('output_test_diy', 'w')
    for i in tqdm(range(test_ch.shape[0] // batch_size)):
        X_batch = test_ch[i * batch_size: i * batch_size + batch_size]
        X_len_batch = len_test_ch[i * batch_size: i * batch_size + batch_size]
        Y_batch = test_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = len_test_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = [l - 1 for l in Y_len_batch]

        feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
        ids = sess.run(sample_id, feed_dict=feed_dict)  # seq_len, batch_size, beam_width
        ids = np.transpose(ids, (1, 2, 0))  # batch_size, beam_width, seq_len
        ids = ids[:, 0, :]  # batch_size, seq_len

        for j in range(ids.shape[0]):
            sentence = translate(ids[j])
            fw.write(sentence + '\n')
    fw.close()

    from nmt.utils.evaluation_utils import evaluate

    for metric in ['bleu', 'rouge']:
        score = evaluate('data/test.en', 'output_test_diy', metric)
        print(metric, score / 100)