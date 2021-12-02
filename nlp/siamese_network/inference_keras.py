from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import pickle

with open('data.pkl', 'rb') as fr:
    data = pickle.load(fr)
    word2id = data['word2id']
    id2word = data['id2word']

train_df = pd.read_csv('train.csv')

stops = set(stopwords.words('english'))


def preprocess(text):
    # input: 'Hello are you ok?'
    # output: ['Hello', 'are', 'you', 'ok', '?']
    text = str(text)
    text = text.lower()

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)  # 去掉其他符号
    text = re.sub(r"what's", "what is ", text)  # 缩写
    text = re.sub(r"\'s", " is ", text)  # 缩写
    text = re.sub(r"\'ve", " have ", text)  # 缩写
    text = re.sub(r"can't", "cannot ", text)  # 缩写
    text = re.sub(r"n't", " not ", text)  # 缩写
    text = re.sub(r"i'm", "i am ", text)  # 缩写
    text = re.sub(r"\'re", " are ", text)  # 缩写
    text = re.sub(r"\'d", " would ", text)  # 缩写
    text = re.sub(r"\'ll", " will ", text)  # 缩写
    text = re.sub(r",", " ", text)  # 去除逗号
    text = re.sub(r"\.", " ", text)  # 去除句号
    text = re.sub(r"!", " ! ", text)  # 保留感叹号
    text = re.sub(r"\/", " ", text)  # 去掉右斜杠
    text = re.sub(r"\^", " ^ ", text)  # 其他符号
    text = re.sub(r"\+", " + ", text)  # 其他符号
    text = re.sub(r"\-", " - ", text)  # 其他符号
    text = re.sub(r"\=", " = ", text)  # 其他符号
    text = re.sub(r"\'", " ", text)  # 去掉单引号
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)  # 把30k等替换成30000
    text = re.sub(r":", " : ", text)  # 其他符号
    text = re.sub(r" e g ", " eg ", text)  # 其他词
    text = re.sub(r" b g ", " bg ", text)  # 其他词
    text = re.sub(r" u s ", " american ", text)  # 其他词
    text = re.sub(r"\0s", "0", text)  # 其他词
    text = re.sub(r" 9 11 ", " 911 ", text)  # 其他词
    text = re.sub(r"e - mail", "email", text)  # 其他词
    text = re.sub(r"j k", "jk", text)  # 其他词
    text = re.sub(r"\s{2,}", " ", text)  # 将多个空白符替换成一个空格

    return text.split()


malstm = load_model('malstm.h5')
correct = 0
for i in range(5):
    print('Testing Case:', i + 1)
    random_sample = dict(train_df.iloc[np.random.randint(len(train_df))])
    left = random_sample['question1']
    right = random_sample['question2']
    print('Origin Questions...')
    print('==', left)
    print('==', right)

    left = preprocess(left)
    right = preprocess(right)
    print('Preprocessing...')
    print('==', left)
    print('==', right)

    left = [word2id[w] for w in left if w in word2id]
    right = [word2id[w] for w in right if w in word2id]
    print('To ids...')
    print('==', left, [id2word[i] for i in left])
    print('==', right, [id2word[i] for i in right])

    left = np.expand_dims(left, 0)
    right = np.expand_dims(right, 0)
    maxlen = max(left.shape[-1], right.shape[-1])
    left = pad_sequences(left, maxlen=maxlen)
    right = pad_sequences(right, maxlen=maxlen)

    print('Padding...')
    print('==', left.shape)
    print('==', right.shape)

    pred = malstm.predict([left, right])
    pred = 1 if pred[0][0] > 0.5 else 0
    print('True:', random_sample['is_duplicate'])
    print('Pred:', pred)
    if pred == random_sample['is_duplicate']:
        correct += 1
print(correct / 5)