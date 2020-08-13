import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.vector', binary=False)

#print the word vector associated with the word
print(model["中国"])

# print top 10 most similar words to the input word
print(model.most_similar(positive='广东', negative=None, topn=10, restrict_vocab=None, indexer=None))

# compare similarities between sentences
sent1 = ['中国','科技','发展','要','靠','人才']
sent2 = ['美国','经济','增长','速度','提升','快']
sent3 = ['我','今天','快乐','生活','快乐','学习']
sim1 = model.n_similarity(sent1, sent2)
sim2 = model.n_similarity(sent1, sent3)
sim3 = model.n_similarity(sent2, sent3)
print("similarity between", sent1, sent2, sim1)
print("similarity between", sent1, sent3, sim2)
print("similarity between", sent2, sent3, sim3)
