#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy 
print(numpy.__version__)


# In[ ]:


import keras 
print(keras.__version__)


# In[ ]:





# In[51]:


with open("fra.txt", 'rt', encoding="utf-8") as f:
    text = f.read()
text[:80]


# In[52]:


lines = text.strip().split('\n')
lines_pairs = [line.split('\t') for line in  lines]
lines_pairs[:15]


# In[53]:


pairs_len = len(lines_pairs)
eng_pair_lens = [len(line_pair[0]) for line_pair in lines_pairs]
fra_pair_lens = [len(line_pair[1]) for line_pair in lines_pairs]
print("一共有{}个法英文语料库对；".format(pairs_len))
print("其中法文语句最短的长度有{}，最长的长度有{}；".format(min(fra_pair_lens), max(fra_pair_lens)))
print("其中英文语句最短的长度有{}，最长的长度有{}。".format(min(eng_pair_lens), max(eng_pair_lens)))


# In[58]:


import re
import numpy as np
from unicodedata import normalize
import string

re_print = re.compile('[^{}]'.format(re.escape(string.printable)))
english_table = str.maketrans('', '', string.punctuation)
cleaned_pairs = list()
for pair in lines_pairs:
    clean_pair = list()
    for i, line in enumerate(pair):
          line = normalize('NFD', line).encode('ascii', 'ignore')
          line = line.decode('UTF-8')
          line = line.split()  
          line = [word.lower() for word in line] 
          line = [word.translate(english_table) for word in line] 
          line = [re_print.sub('', w) for w in line] 
          line = [word for word in line if word.isalpha()] 
          clean_pair.append(' '.join(line))
    cleaned_pairs.append(clean_pair) 
cleaned_pairs = np.array(cleaned_pairs) 
cleaned_pairs[:15]


# In[61]:


print(string.printable)
print(string.punctuation)


# In[ ]:


import pickle
with open("french_to_english.pkl", "wb") as f:
    pickle.dump(cleaned_pairs, f)


# In[63]:


with open("french_to_english.pkl", "rb") as f:
    raw_dataset = pickle.load(f)

sequence_length = 10000
dataset = raw_dataset[:sequence_length]
np.random.shuffle(dataset)
dataset[:15]


# In[64]:


train_len = sequence_length - 1500
train, test = dataset[:train_len], dataset[train_len:]

def save_dataset(sentences, filename):
    with open(filename, 'wb') as f:
        pickle.dump(sentences, f)

save_dataset(dataset, "french_to_english_dataset_top10000.pkl")
save_dataset(train, "french_to_english_train.pkl")
save_dataset(test, "french_to_english_test.pkl")

print("train.shape={}, test.shape={}".format(train.shape, test.shape))


# In[ ]:


print(train)


# In[65]:


from keras.preprocessing.text import Tokenizer

texts = ['I love AI in China', '特拉字节', 'AI 人工智能']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print("tokenizer.word_index={}.".format(tokenizer.word_index))
print("tokenizer.texts_to_sequences={}.".format(tokenizer.texts_to_sequences(texts)))


# In[72]:


from keras.preprocessing.text import Tokenizer

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('英文序列单词最大个数{}，单词有{}个。'.format(eng_length, eng_vocab_size))

fra_tokenizer = create_tokenizer(dataset[:, 1])
fra_vocab_size = len(fra_tokenizer.word_index) + 1
fra_length = max_length(dataset[:, 1])
print('法文序列单词最大个数{}，单词有{}个。'.format(fra_length, fra_vocab_size))


# In[ ]:





# In[ ]:


from keras.preprocessing import sequence
from keras import utils

def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = sequence.pad_sequences(X, maxlen=length, padding='post')
    return X

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = utils.to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

X_train = encode_sequences(fra_tokenizer, fra_length, train[:, 1])
y_train = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
y_train = encode_output(y_train, eng_vocab_size)
 
X_test = encode_sequences(fra_tokenizer, fra_length, test[:, 1])
y_test = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
y_test = encode_output(y_test, eng_vocab_size)


# In[77]:


from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed

def create_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
 
model = create_model(fra_vocab_size, eng_vocab_size, fra_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()


# In[ ]:


from keras import backend as K 
K.clear_session()


# In[78]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

callbacks_EarlyStopping = EarlyStopping(monitor='val_loss', 
                                        patience=3)

model_filename = 'translator_weights_model.h5'
checkpoint_ModelCheckpoint = ModelCheckpoint(model_filename, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')
history = model.fit(X_train, 
                    y_train, 
                    epochs=50, 
                    batch_size=64, 
                    validation_data=(X_test, y_test), 
                    callbacks=[checkpoint_ModelCheckpoint, callbacks_EarlyStopping], 
                    verbose=2)


# In[ ]:


history.history.keys()


# In[ ]:


def load_clean_sentences(filename):
    return pickle.load(open(filename, 'rb'))

dataset = load_clean_sentences('french_to_english_dataset_top10000.pkl')
train_ds = load_clean_sentences('french_to_english_train.pkl')
test_ds = load_clean_sentences('french_to_english_test.pkl')

eng_tokenizer = create_tokenizer(dataset[:, 0])
fra_tokenizer = create_tokenizer(dataset[:, 1])
fra_length = max_length(dataset[:, 1])

X_train = encode_sequences(fra_tokenizer, fra_length, train_ds[:, 1])
X_test = encode_sequences(fra_tokenizer, fra_length, test_ds[:, 1])


# In[ ]:


from keras import models
model = models.load_model('translator_weights_model.h5')


# In[ ]:


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# In[92]:


from nltk.translate.bleu_score import corpus_bleu

def test_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('源语句=[{}], 目标语句=[{}], 预测语句=[{}]'.format(raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    print('BLEU-1: {}'.format(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))))
    print('BLEU-2: {}'.format(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))))
    print('BLEU-3: {}'.format(corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))))
    print('BLEU-4: {}'.format(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))))

print('训练集：')
test_model(model, eng_tokenizer, X_train, train_ds)
print('测试集：')
test_model(model, eng_tokenizer, X_test, test_ds)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




