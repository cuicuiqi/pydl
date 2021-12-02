
from textgenrnn import textgenrnn

textgen = textgenrnn()
# 从文件中训练模型
textgen.train_from_file('Simpsons_Episode_18.txt', num_epochs=30)
# 根据Temperature的值来生成剧本文本
textgen.generate()
# 生成10行文本
textgen.generate_to_file('my_generated_texts.txt', n=10)


