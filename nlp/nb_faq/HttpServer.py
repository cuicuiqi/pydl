import socket
import urllib.parse
from threading import Thread
from gensim.models import Word2Vec
from gensim import models
from gensim import corpora
import CustomTextPreOps
import FaqMySQL
import numpy as np
from threading import Timer
import logging
import traceback

logging.basicConfig(level=logging.INFO,
                    filename='nlp.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def handle_request(conn,stopWords, vecModel, tfidfModel, dictionary, vecfaqTuples):
    response_start_line = "HTTP/1.1 200 OK\r\n"
    response_headers = "Server: NBAPI python server\r\n\r\n"

    try:
        request = conn.recv(1024)

        logging.info('received:' + str(request.decode()))

        path = request.decode().split(' ')[1]
        if (path == '/'):
            conn.send(bytes(response_start_line + response_headers, "utf-8"))
            return

        data = path.split('/')[2]
        data = urllib.parse.unquote(data)
        nparr = CustomTextPreOps.preOps(data, stopWords, vecModel, tfidfModel, dictionary)

        logging.info(nparr)

        if nparr is None :
            raise RuntimeError('not valid input word!')

        bestCSValue = 0
        bestTuple = ()
        for vecfaqTuple in vecfaqTuples:
            vec = vecfaqTuple[0]
            csValue = cos_sim(nparr, vec)
            if csValue > bestCSValue:
                bestCSValue = csValue
                bestTuple = (vecfaqTuple[1], vecfaqTuple[2])


        response_body = str(bestTuple)
        response = response_start_line + response_headers + response_body
        conn.send(bytes(response, "utf-8"))
    except Exception as e:
        logging.error("handler thread:" + str(e))
        logging.error(traceback.format_exc())
        response_body = str(e)
        response = response_start_line + response_headers + response_body
        conn.send(bytes(response, "utf-8"))
    finally:
        conn.close()
        logging.info('response done.')



def getVecfaqTuples(stopWords, vecModel, tfidfModel, dictionary):
    faqs = FaqMySQL.getAllFAQ()
    vecfaqTuples = []
    for k, v in faqs.items():
        nparr = CustomTextPreOps.preOps(v, stopWords, vecModel, tfidfModel, dictionary)
        vecfaqTuples.append((nparr, k, v))

    for t in vecfaqTuples:
        logging.info(t)
    return vecfaqTuples


def timerUpdate(vecfaqTuples, stopWords, vecModel, tfidfModel, dictionary):
    try:
        v = getVecfaqTuples(stopWords, vecModel, tfidfModel, dictionary)
        vecfaqTuples.clear()
        vecfaqTuples += v
        logging.info('update faq tuples finished.' + str(len(vecfaqTuples)))
        timer = Timer(7200, timerUpdate, (vecfaqTuples, stopWords, vecModel, tfidfModel, dictionary,))
        timer.start()
    except Exception as e:
        logging.error("timer thread:" + str(e))
        logging.error(traceback.format_exc())


stopWords = CustomTextPreOps.readStopWord()
vecModel = Word2Vec.load('./model.w2v')
tfidfModel = models.TfidfModel.load("./model.tfidf")
dictionary = corpora.Dictionary.load("./model.dict")
vecfaqTuples = []
timerUpdate(vecfaqTuples, stopWords, vecModel, tfidfModel, dictionary)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('',8000))
sock.listen(128)

while True:
    try:
        conn, addr = sock.accept()
        logging.info('connected:' + str(addr))
        handle_client_process = Thread(target=handle_request,
                                       args=(conn, stopWords, vecModel, tfidfModel, dictionary, vecfaqTuples))
        handle_client_process.start()
    except Exception as e:
        logging.error("main thread:" + str(e))
        logging.error(traceback.format_exc())






