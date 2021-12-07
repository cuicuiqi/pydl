import os
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot

cb = ChatBot("Cb")
trainer = ListTrainer(cb)

path = './clean_chat_corpus'
children = os.listdir(path)
for child in children:
    child = os.path.join(path, child)
    fin = open(child, 'r', encoding='utf-8')
    print('begin new file:' + child)
    for line in fin:
        qa = line.split('\t')
        trainer.train(qa)

