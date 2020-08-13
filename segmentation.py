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




# with open('/Users/jiahuichen/Desktop/icwb2-data/training/pku_training.txt',encoding='GBK') as f:
#     data = f.read()
# with open('utf8.txt','w',encoding='utf8') as f:
#     f.write(data)
# f = io.open ('utf8.txt', mode = 'r', encoding= 'utf-8')
# g = io.open('pku_corpus.txt', mode = 'w', encoding = 'utf-8')
# skip_list = ['','。', '，', '、', '：', '“', '”', "《", "》", '\n','；','— —','（', '）']
#
# sentence = f.readline()
# while sentence:
#     for skip_char in skip_list:
#         sentence = sentence.replace(skip_char, '')
#         sentence =sentence.strip()
#         sentence = sentence.replace('    ', ' ')
#         sentence = sentence.replace('  ', ' ')
#     g.write(sentence)
#     g.write('\n')
#     sentence = f.readline()
# f.close()
# g.close()