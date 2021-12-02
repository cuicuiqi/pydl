# 加载包
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 训练模型
sentences = LineSentence('wiki.zh.word.text')
model = Word2Vec(sentences, size=128, window=5, min_count=5, workers=4)

# 保存模型
model.save('word_embedding_128')

# 加载模型
model = Word2Vec.load("word_embedding_128")

# 使用模型
# 相关词
items = model.wv.most_similar('数学')
for i, item in enumerate(items):
	print(i, item[0], item[1])
# 语义类比
print('=' * 20)
items = model.wv.most_similar(positive=['纽约', '中国'], negative=['北京'])
for i, item in enumerate(items):
	print(i, item[0], item[1])
# 不相关词
print('=' * 20)
print(model.wv.doesnt_match(['早餐', '午餐', '晚餐', '手机']))
# 计算相关度
print('=' * 20)
print(model.wv.similarity('男人', '女人'))