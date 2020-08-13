import io
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

a = io.open("sentences.txt", mode="r+", encoding="utf-8")
model = Word2Vec(LineSentence(a), size=400, window = 5, min_count= 5, workers= multiprocessing.cpu_count() - 4)
model.save('word2.model')
model.wv.save_word2vec_format('word2vec.vector', binary=False)