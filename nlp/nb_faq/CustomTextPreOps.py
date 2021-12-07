import jieba
import re
import numpy as np

def readStopWord():
    stop_words = list()
    stopwords = open('./stop_words.txt','r',encoding='utf-8')
    for word in stopwords:
        word = word.strip()
        if len(word) != 0:
            stop_words.append(word)
    stopwords.close()
    return stop_words


def preOps(rawline, stopwords, vecModel, tfidfModel, dictionary):
    line = re.sub(r'[^\u4e00-\u9fa5]', "", rawline)  # 只保留中文
    if len(line) == 0:
        return

    seglist = jieba.cut(line)
    words = []
    id2vec = {}
    length = 0
    for seg in seglist:
        if seg not in stopwords:
            try:
                vec = vecModel[seg]
            except KeyError as e:
                print(e)
                continue

            idx = dictionary.doc2idx([seg])[0]
            if idx == -1:
                continue

            words.append(seg)
            # print(seg + ' ' + str(idx) + ' ' + str(vec))
            id2vec[idx] = vec
            length += 1

    if length <= 0:
        return

    bow = dictionary.doc2bow(words)
    tuplelist = tfidfModel[bow]
    # print(tuplelist)

    sumValue = np.zeros(128)
    for (k,tfidf) in tuplelist:
        vec = id2vec[k]
        sumValue += vec * tfidf

    sumValue = sumValue / length
    return sumValue