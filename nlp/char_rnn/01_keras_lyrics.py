# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import LambdaCallback
import numpy as np
import random
import pickle

sentences = []
with open('../lyrics.txt', 'r', encoding='utf8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        count = 0
        for c in line:
            if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'):
                count += 1
        if count / len(line) < 0.1:
            sentences.append(line)
print('共%d首歌' % len(sentences))

chars = {}
for sentence in sentences:
    for c in sentence:
        chars[c] = chars.get(c, 0) + 1
chars = sorted(chars.items(), key=lambda x:x[1], reverse=True)
chars = [char[0] for char in chars]
vocab_size = len(chars)
print('共%d个字' % vocab_size, chars[:20])

char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([char2id, id2char], fw)

#整理训练数据，定义模型并编译
maxlen = 10
step = 3
embed_size = 128
hidden_size = 128
vocab_size = len(chars)
batch_size = 64
epochs = 20

X_data = []
Y_data = []
for sentence in sentences:
    for i in range(0, len(sentence) - maxlen, step):
        X_data.append([char2id[c] for c in sentence[i: i + maxlen]])
        y = np.zeros(vocab_size, dtype=np.bool)
        y[char2id[sentence[i + maxlen]]] = 1
        Y_data.append(y)
X_data = np.array(X_data)
Y_data = np.array(Y_data)
print(X_data.shape, Y_data.shape)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=maxlen))
model.add(LSTM(hidden_size, input_shape=(maxlen, embed_size)))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


#定义每轮训练结束后的回调函数
def on_epoch_end(epoch, logs):
    print('-' * 30)
    print('Epoch', epoch)

    index = random.randint(0, len(sentences))
    for diversity in [0.2, 0.5, 1.0]:
        print('----- diversity:', diversity)
        sentence = sentences[index][:maxlen]
        print('----- Generating with seed: ' + sentence)
        sys.stdout.write(sentence)

        for i in range(400):
            x_pred = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                x_pred[0, t] = char2id[char]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = id2char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()


#训练
model.fit(X_data, Y_data, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
model.save('song_keras.h5')




#调用模型生成歌词，需提供一句起始歌词
# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import pickle
import sys

maxlen = 10
model = load_model('song_keras.h5')

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char] = pickle.load(fr)

def sample(preds, diversity=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

sentence = '能不能给我一首歌的时间'
sentence = sentence[:maxlen]

diversity = 1.0
print('----- Generating with seed: ' + sentence)
print('----- diversity:', diversity)
sys.stdout.write(sentence)

for i in range(400):
    x_pred = np.zeros((1, maxlen))
    for t, char in enumerate(sentence):
        x_pred[0, t] = char2id[char]

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = id2char[next_index]

    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()




