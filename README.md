---
title: How I Created Chinese Word Embeddings with Word2vec
date: 2020-08-11 15:40:42
tags:
cover: https://pic.imgdb.cn/item/5f3271f214195aa594a74bb9.jpg
---
## Intro
Word2Vec is a common tool in Natural Language Processing (NLP) for generating word embeddings from text. Word embeddings are essentially vectors in a higher dimensional space - by translating words into these vectors, we can quantitatively represent the words within their context. The hope is to group similar words together just by looking at their context. In this blog post, I will walk through how I used the Word2Vec model to generate word embeddings using Gensim’s Word2Vec model. I will also show some of the applications of the model. 

## Word2Vec
### Intro to Word2Vec
Word2Vec is a tool for Natural Language Processing - how to make machine “understand” human language. This task is incredibly difficult, as human languages are extremely complex, with unstructured expressions and weird combinations of words and slurs. One common example used to illustrate the weirdness of Chinese language is this joke:
> 阿呆给领导送红包,领导说:你这是什么意思? 阿呆:意思意思. 领导:你这就不够意思了. 阿呆:小意思 ,小意思. 领导:你这人真有意思. 阿呆:没有别的意思. 领导:那我就不好意思了

In this conversation, the Chinese word “意思” appeared numerous times, each with a different meaning. It is already difficult for someone without extensive experience living in China to pick up the meaning of this conversation, not to mention making machine understand it. 
Then how can we let the machine learn the semantics of our weird and complex languages? The solution that Word2Vec comes up with is to use context. The idea behind it is best summarized by this maxim:
> Context is the key – from that comes the understanding of everything.

This may seem pretty intuitive but it is actually a very brilliant idea. Machine does no necessarily need to understand the semantics - all it needs to do is to infer whether it is common for a word to be at where it is given the context, which can be accomplished using a neural network. 
![][image-1]
I will not go into details about how a neural network functions. If you are interested, I highly recommend [this video]() by 3Blue1Brown - the visualization helped me understand the concept a lot. 

In the neural network for Word2Vec, the inputs is an one-hot vector. The hidden layer does not have activation function, meaning it uses linear activation. The output layer is produced using Softmax - a classifier commonly used in machine learning task. It is important to note that what we want is not necessarily to use the model to accomplish any task; instead, we will take the parameters of the model in the form of a matrix - where each row would be a word vector that we want as result. 

### How Does the Training Work
That being said, there are commonly two ways to train the word2vec model, and word2vec uses both ways
- CBOW (Continuous Bag-of-Words) - where we use the context to infer the word
- Skip Gram - where we use the word to infer the context
I will briefly talk about how the training functions with an example using the skip gram method. 
Suppose the sentence that we want to train the model with is 
> The quick brown fox jumps over the lazy dog.

At this point, we need to specify our skip window, which is the window of context that we want the machine to predict. Suppose we set that to 2, then we can generate the following training data set
![][image-2]
We want to turn this into a classifier problem, so we turn the data set into one-hot encoding, generating a series of vectors where 1 appears on where the word is and 0 appears on everywhere else
![][image-3]
We put the training data into the neural network, and use back propagation to minimize the loss function,  hoping to make the resulting vector as close to our desired output. When the training is completed, we take out the embedding matrix, which will contain all the word vectors we want. 
![][image-4]

## Getting and Processing Data
In the following sections, I will walk through how I trained a Word2Vec model using data from Peking University’s Chinese NLP data set. 
I downloaded my dataset from [Kaggle]() - a popular data science website containing a lot of useful dataset. My data consist of transcripts of Xinwenlianbo (新闻联播), an every-day TV news broadcast -  over the past few years. 
Below is my code for processing the data and cutting them into individual words 
```python
import pandas as pd
import io
import jieba
import re

# this would be the output file 
f = io.open('sentences.txt', mode = 'w', encoding= 'utf-8')

# read the csv file using pandas
news = pd.read_csv('chinese_news.csv')

# take out all the news body 
x = list(news['content'])

l = []

# I want to skip all Chinese punctuactions
skip_list = ['','。', '，', '、', '：', '“', '”', "《", "》", '\n','；','— —','（', '）']

# looping through all the news stories, col would store individual news stories
for col in x:
    # take the text out and remove white space
    text = str(col).rstrip()
    # split the text by sentence
    sentences = re.split('。|！|\!|\.|？|\?', text)
    for i in range(int(len(sentences) / 2)):
        sentence = sentences[2 * i] + sentences[2 * i + 1]
        # skip all the punctuation
        for skip_char in skip_list:
            sentence = sentence.replace(skip_char, '')
        # I use jieba, a Chinese language segmentation tool, to cut the sentence into words
        seg_list = list(jieba.cut(sentence))
        for temp_term in seg_list:
            l.append(temp_term)
        # output format would be words seperated by white spaces
        res = " ".join(l) + '\n'
        res.lstrip()
        # write the result into output file
        f.write(res)
        l = []

f.close()
```
One thing to note is that I used jieba for word segmentation. You can find extensive instructions on how to pip install it online. Another common tool is hanlp (which is probably more advanced too). For simplicity I sticked to jieba.
After the program is run the output file should look like this
![][image-5]
## Using the Gensim Word2Vec Model
The use of Genism Word2Vec Model is fairly simple and straight-forward after pre-processing. You can also find instructions on how to pip install gensim online - it is quite easy from my experience. 
```python
import io
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

a = io.open("sentences.txt", mode="r+", encoding="utf-8")
model = Word2Vec(LineSentence(a), size=400, window = 5, min_count= 5, workers= multiprocessing.cpu_count() - 4)
model.save('word2.model')
model.wv.save_word2vec_format('word2vec.vector', binary=False)
```
As you can see, there are several parameters you can choose when training the model
- size: the dimension of output vector. In most cases higher dimensions correlate with increased accuracy - the intuition behind that more dimension means the vector can store more nuanced relations. I set it to 400, which is the recommended default.
- window: the size of window that you want to use during training. That is, how many words do you want the machine to be looking at beyond the current word. A larger window means the algorithm is considering more context and vice versa.
- min count: this is used in the initial processing. Words with frequency less than the min count would be discarded before the training begins.
- workers: how many cpu workers you want to use in the training. 

After the training, I stored the model in two formats
- .model: models stored in this format can be retrained, but the format can not be decoded for you to see
- .vector: models stored in this format cannot be retrained, but the format can be decoded for you to see
My .vector file looks like this - you can see each word is associated with a word vector
![][image-6]
## Applying the model
The last step would be to use the model to do something. As it turns out Gensim lets you do a lot of things with your model. Below are some examples
```python
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

```
The output looks like this 
```python
[-0.05930993 -0.60154545 -0.19779043 -0.48194197 -0.30343688 -1.0903504
  1.7381277  -1.1709697   0.2171865  -0.59634805  0.6168709   0.36309278
 -0.00500764  0.51317024 -0.17959161 -0.8421643  -0.7322384   0.689643
  0.31742564  0.65299606 -0.43274835  0.26696515  0.41932794  0.32154498
  1.3054777   1.3687644   0.6937966  -0.58396184 -0.35994187  1.5102268
 -0.3021141   0.35180223  0.98756677 -0.6793006  -0.8039679  -0.11551007
  0.23520286 -0.6949743  -0.4775846   0.62763804  0.32157004  0.10934389
 -0.83816004 -1.005785   -0.26850745  0.7644115   0.8944372  -1.192454
  0.1794393  -0.5560042   0.11054779 -0.2966826   0.7419602   0.7677035
  0.13389288  0.15937632 -0.11777601  0.12296207 -0.64838636 -0.5888581
 -0.2861753   1.6017598   0.6768199  -1.1618366   0.9654343   1.6714641
  0.5946807   0.41378683 -1.1309861  -0.5350951  -0.61663336 -0.74870783
 -0.6871212   0.17910142  0.7646165   0.3806211  -0.5449704   0.5622143
 -0.97382987  0.5746229   0.985208    1.2262727  -0.9806304   1.2106696
 -0.3302971   0.27300367  0.06429467  0.48375     0.521364    0.9094922
  0.11546493  0.09551446  0.16652733  0.03306641 -0.07538961 -1.0116953
 -0.13459866  1.8211585  -0.6950327   1.2637784  -0.52912796 -0.04460784
  1.0214093   0.10162195 -0.91459376 -0.17390226 -0.17388315 -0.2754795
  0.20606285  0.15042375  0.13845378  0.02339607 -1.5108862  -0.45250568
 -0.9359936   0.49126223  1.1190046  -0.87515837 -0.57949454  0.6871173
 -0.12499838  1.9581158   0.95917594 -0.10627772  0.54702    -0.9470129
 -1.6031312  -0.07724522 -0.05718271  0.5277461  -0.06041054 -0.23856659
  0.5542616  -1.5007663  -0.2780823  -0.2115478  -1.4245342  -0.01399347
  0.60007745 -0.01684805 -1.6818323  -0.1931889  -1.5237374  -1.4075873
  0.02254261 -0.5751521  -0.62415326  0.3235581   0.06705747 -1.0198073
 -0.6251598  -0.91364276 -0.176965   -0.06620737  1.8866454  -0.24774651
 -0.21445335  1.1362627   1.273662    1.3616595  -1.2018119   1.4867073
 -0.07305669  0.16995254  0.70268124  0.04884331  0.04264761 -0.24667963
 -0.03846245 -0.09678071 -0.27120808 -0.05220251  0.08652628  0.49769568
 -0.5352079   0.46929055  0.6785873   0.49454334 -1.4178201   0.4647789
  1.2762723  -1.1194816  -1.010007   -0.6022205   0.6323132  -0.16102295
  1.1645333   0.9596903  -0.51386905  0.588219    0.08259875  0.3908056
  1.0106333  -0.14321563  0.9018996  -1.0544982  -0.23611164  1.0572827
  0.18362053  0.39346454  1.0405265  -0.5581456   0.17081226 -0.24262072
 -0.15824328 -0.5129465  -0.20730579  0.49548322 -0.7830268   1.1653224
  0.28673372  1.5712413  -0.9241767  -1.150959   -0.58597785 -0.6976993
  0.8195765   0.88995826 -0.53419846 -0.30637687 -0.6004372  -0.236429
 -0.7772929  -0.03667458  0.45034775 -0.67187023 -0.4030743  -0.8062104
  2.623117    0.02814003  0.00678085  0.47618607 -1.2426374   1.2759159
  0.56348497 -0.0306225   0.07943363  0.33734903  0.74658936 -0.74653673
 -0.6815518  -0.06754778 -1.8862988  -1.723702    0.39478    -0.3838985
 -0.24430585 -0.87098897  0.12656392  0.3491659  -0.06678118  0.8528555
  0.62753266  0.80654687 -0.10528732 -0.06510979  0.60498166  1.1265533
  0.12893651  0.22323228 -0.43933755  0.30504924  0.27831098 -0.9089467
 -0.25216293  0.45882642  1.0253651   0.83845913  0.19217832 -0.25366136
 -1.1501366  -0.59544957  0.71397233 -0.33581296 -0.3301465   1.0448421
 -0.98036987 -0.5961618   1.4408491  -0.89590365 -1.4903147   0.48930526
 -0.20611231  0.80190253 -1.17446    -0.44388193 -0.864709    0.7994709
  0.6211015   1.0785302  -0.39803782 -0.06914762 -1.2612135  -0.43342596
  0.9597424  -0.4313669  -0.83414686  0.40143782  0.085173   -0.1414892
 -0.244692    1.5626423  -0.04647842  1.294185    0.70646983 -0.93281174
  1.7463619  -1.5049967   1.2497619  -0.4949764  -1.1369902   0.583246
 -1.9625047  -1.9645227   0.39969036  0.1845216  -2.224353    0.36199805
 -1.4422947  -1.8587517   1.0950716   1.6715609  -1.8970809   1.2119989
 -0.9918985   0.2168934  -0.03940273 -1.4473706  -0.1715694   0.40139624
 -0.20749897 -0.23848538  0.49070346  1.8710481  -0.22102888  1.8875043
  0.1575313  -2.143685    0.9722614  -0.18779267 -1.5262072   0.01395513
  0.6219663   0.6200599   1.0746591  -0.6949839  -0.11403554 -0.11398692
 -1.1922437  -0.6713476  -0.04333466  1.6420592   0.21196647 -0.6093193
 -0.13868634  0.8807369   0.5370432  -1.0475727  -0.38448784 -0.03109699
 -0.665416    0.11490192 -0.46147627  1.2118089   0.78576756  0.7781474
  0.13457988  0.0555984   0.01180293  0.69014937  0.44752008  1.715098
  0.10089613 -0.8013588   0.53294414  0.12885724  0.3112086   1.6809343
  0.00383066 -0.9831795   1.19722    -1.5555032   0.3774249  -0.8096407
  0.5436396  -0.33276993 -1.1977743  -0.2084381  -0.9183242  -0.37455353
 -1.7807964   0.5187864   0.22164525  0.48849887  0.7664638  -0.46407062
 -0.7822278  -0.12114441  0.8138251  -0.6337241 ]
[('福建', 0.9159765243530273), ('湖北', 0.9119961261749268), ('江苏', 0.9061319828033447), ('安徽', 0.8997495174407959), ('陕西', 0.8922166228294373), ('广西', 0.8836355209350586), ('河南', 0.8768578171730042), ('浙江', 0.8689074516296387), ('江西', 0.868665337562561), ('山东', 0.8685473203659058)]
similarity between ['中国', '科技', '发展', '要', '靠', '人才'] ['美国', '经济', '增长', '速度', '提升', '快'] 0.4123175
similarity between ['中国', '科技', '发展', '要', '靠', '人才'] ['我', '今天', '快乐', '生活', '快乐', '学习'] 0.30470887
similarity between ['美国', '经济', '增长', '速度', '提升', '快'] ['我', '今天', '快乐', '生活', '快乐', '学习'] 0.05903372

Process finished with exit code 0
```
The results are pretty good - when I try to output the most similar words to “广东” (Guangdong), all the words are similar Chinese provinces. Also Word2Vec has broader applications than this. Once you have all the word embeddings, you can easily compute similarities between sentences, paragraphs, and even texts. Indeed, Word2Vec was used in a variety of NLP tasks - classification, emotion detection, name entity recognition (NER). The reason I said it was is because it has been replaced by more advanced models like BERT. However, Word2Vector has been a foundation of many NLP models including BERT, so understanding the mechanisms of this model would definitely help you take your first step in NLP. 





[image-1]:	https://pic.imgdb.cn/item/5f327e5014195aa594ac4bac.jpg
[image-2]:	https://pic.imgdb.cn/item/5f327e9614195aa594ac6a6b.jpg
[image-3]:	https://pic.imgdb.cn/item/5f327eb514195aa594ac75ef.jpg
[image-4]:	https://pic.imgdb.cn/item/5f327ed114195aa594ac7fec.jpg
[image-5]:	file:///.file/id=6571367.21844725
[image-6]:	https://pic.imgdb.cn/item/5f32b55614195aa594bf8845.png
