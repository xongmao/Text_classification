# 引言
采用传统机器学习的方法词袋模型+tfidf+朴素贝叶斯/SVM，得到分类结果作为参考。  
再利用 word2vec 训练词向量，用深度学习模型 cnn/rnn/attn 等进行文本分类，以及  
用FastText 模型训练词向量并进行文本分类。
# 数据集
用于分类文本的数据使用经典的 20 类新闻包，里面大约有 20000 条新闻，比较均衡地  
分成了20 类。数据来源信息(Collected from UseNet postings over a period of several  
months in 1993)。词向量的训练采用 Mikolov 曾经使用过的 text8 数据包进行训练，以及  
使用nltk库对数据进行预处理。

# 结果
训练好的模型分别保存在cnn, rnn, attn文件夹中。  
tfidf+朴素贝叶斯：得到的分类结果的准确率为0.916。  
tfidf+SVM：得到的分类结果的准确率为 0.919。  
word2vec+cnn：得到的分类结果的准确率为 0.905。  
word2vec+rnn：得到的分类结果的准确率为 0.894。      
word2vec+attn：得到的分类结果的准确率为 0.890。  
模型融合(cnn+rnn+attn)：得到的分类结果的准确率为 0.917。  
fasttext：得到的分类结果的准确率为 0.884。
