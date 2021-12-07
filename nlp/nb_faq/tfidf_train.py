from gensim import models
from gensim import corpora

def readCombineCorpus():
    all_corpus = []
    o1 = open('./corpus1_out.txt','r',encoding='utf-8')
    for line in o1:
        cutlast = line.split(' ')[0:-1]
        all_corpus.append(cutlast)
    o1.close()

    o2 = open('./corpus2_out.txt', 'r', encoding='utf-8')
    for line in o2:
        cutlast = line.split(' ')[0:-1]
        all_corpus.append(cutlast)
    o2.close()

    o3 = open('./corpus3_out.txt', 'r', encoding='utf-8')
    for line in o3:
        cutlast = line.split(' ')[0:-1]
        all_corpus.append(cutlast)
    o3.close()

    return all_corpus


# 赋给语料库中每个词(不重复的词)一个整数id
all_corpus = readCombineCorpus()
dictionary = corpora.Dictionary(all_corpus)
dictionary.save("./model.dict")

new_corpus = [dictionary.doc2bow(text) for text in all_corpus]
tfidf = models.TfidfModel(new_corpus)
tfidf.save("./model.tfidf")



