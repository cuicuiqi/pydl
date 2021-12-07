#!/usr/bin/env python
# coding: utf-8

# ### 命名实体识别

# ![title](1.png)

# ### 网络模型
# 

# ![title](2.png)

# In[ ]:





# In[ ]:


import numpy as np
from sklearn.model_selection import ShuffleSplit
from data_utils import ENTITIES, Documents, Dataset, SentenceExtractor, make_predictions
from data_utils import Evaluator
from models import build_lstm_crf_model
from gensim.models import Word2Vec


# ### 瑞金医院数据集

# In[2]:


data_dir = 'ruijin_round1_train2_20181022/'
ent2idx = dict(zip(ENTITIES, range(1, len(ENTITIES) + 1)))
idx2ent = dict([(v, k) for k, v in ent2idx.items()])


# In[3]:


docs = Documents(data_dir=data_dir)
rs = ShuffleSplit(n_splits=1, test_size=20, random_state=2018)
train_doc_ids, test_doc_ids = next(rs.split(docs))
train_docs, test_docs = docs[train_doc_ids], docs[test_doc_ids]


# In[4]:


train_docs[0]


# In[5]:


num_cates = max(ent2idx.values()) + 1
sent_len = 64
vocab_size = 3000
emb_size = 100
sent_pad = 10
sent_extrator = SentenceExtractor(window_size=sent_len, pad_size=sent_pad)
train_sents = sent_extrator(train_docs)
test_sents = sent_extrator(test_docs)

train_data = Dataset(train_sents, cate2idx=ent2idx)
train_data.build_vocab_dict(vocab_size=vocab_size)

test_data = Dataset(test_sents, word2idx=train_data.word2idx, cate2idx=ent2idx)
vocab_size = len(train_data.word2idx)


# In[6]:


w2v_train_sents = []
for doc in docs:
    w2v_train_sents.append(list(doc.text))
w2v_model = Word2Vec(w2v_train_sents, size=emb_size)

w2v_embeddings = np.zeros((vocab_size, emb_size))
for char, char_idx in train_data.word2idx.items():
    if char in w2v_model.wv:
        w2v_embeddings[char_idx] = w2v_model.wv[char]


# In[7]:


seq_len = sent_len + 2 * sent_pad
model = build_lstm_crf_model(num_cates, seq_len=seq_len, vocab_size=vocab_size, 
                             model_opts={'emb_matrix': w2v_embeddings, 'emb_size': 100, 'emb_trainable': False})
model.summary()


# In[8]:


train_X, train_y = train_data[:]
print('train_X.shape', train_X.shape)
print('train_y.shape', train_y.shape)


# ### 训练模型

# In[9]:


model.fit(train_X, train_y, batch_size=64, epochs=10)


# ### 测试结果

# In[10]:


test_X, _ = test_data[:]
preds = model.predict(test_X, batch_size=64, verbose=True)
pred_docs = make_predictions(preds, test_data, sent_pad, docs, idx2ent)


# In[11]:


f_score, precision, recall = Evaluator.f1_score(test_docs, pred_docs)
print('f_score: ', f_score)
print('precision: ', precision)
print('recall: ', recall)


# In[12]:


sample_doc_id = list(pred_docs.keys())[0]
test_docs[sample_doc_id]


# In[13]:


pred_docs[sample_doc_id]

