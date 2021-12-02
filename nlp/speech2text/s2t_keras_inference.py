# -*- coding: utf-8 -*-

from keras.models import load_model
from keras import backend as K
import numpy as np
import librosa
from python_speech_features import mfcc
import pickle
import glob

wavs = glob.glob('data/*.wav')
with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

mfcc_dim = 13
model = load_model('asr.h5')

index = np.random.randint(len(wavs))
print(wavs[index])

audio, sr = librosa.load(wavs[index])
energy = librosa.feature.rmse(audio)
frames = np.nonzero(energy >= np.max(energy) / 5)
indices = librosa.core.frames_to_samples(frames)[1]
audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
X_data = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
X_data = (X_data - mfcc_mean) / (mfcc_std + 1e-14)
print(X_data.shape)

with open(wavs[index] + '.trn', 'r', encoding='utf8') as fr:
    label = fr.readlines()[0]
    print(label)

pred = model.predict(np.expand_dims(X_data, axis=0))
pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
pred_ids = pred_ids.flatten().tolist()
print(''.join([id2char[i] for i in pred_ids]))