import jieba
import os
import re

def splitSentence(folder, outputFile, stopwords):
    if not os.path.isdir(folder):
        return

    children = os.listdir(folder)
    for child in children:
        child = os.path.join(folder, child)
        if os.path.isdir(child):
            splitSentence(child, outputFile, stopwords)
        else:
            readFile(child, outputFile, stopwords)


def readFile(path, outputFile, stopwords):
    try:
        fin = open(path, 'r', encoding='utf-8')

        for line in fin:

            line = re.sub(r'[^\u4e00-\u9fa5]', "", line) #只保留中文
            if len(line) == 0:
                continue

            seglist = jieba.cut(line)
            outstr = ''
            length = 0
            for seg in seglist:
                if seg not in stopwords:
                    outstr += seg + ' '
                    length += 1

            if length <= 1:
                continue

            outputFile.write(outstr + '\n')
    except:
        print(path)
    finally:
        fin.close()

def readStopWord():
    stop_words = list()
    stopwords = open('./stop_words.txt','r',encoding='utf-8')
    for word in stopwords:
        word = word.strip()
        if len(word) != 0:
            stop_words.append(word)
    stopwords.close()
    return stop_words


# fout = open('./corpus1_out.txt','w',encoding='utf-8')
# splitSentence('./corpus1',fout, readStopWord())
# fout.close()

# fout = open('./corpus2_out.txt','w',encoding='utf-8')
# splitSentence('./corpus2',fout, readStopWord())
# fout.close()

fout = open('./corpus0_out.txt','w',encoding='utf-8')
splitSentence('./corpus0',fout, readStopWord())
fout.close()