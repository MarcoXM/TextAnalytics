#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from nltk.tokenize import regexp_tokenize
from __future__ import print_function
from gensim.models import KeyedVectors
from gensim.models import word2vec
from gensim.scripts import glove2word2vec

## getting the tokens in sample data


def news2vec(vocabulary, pretrainVec, pretrainVocab):
    '''
    @ vocabulary : list of words in news,
    @ pretrainVec : imported pretain-model data
    @ pretrainVocab : list of words in pretrain Vecter。
    '''
    import collections
    key_word2vec = collections.defaultdict(list)
    for w in vocabulary:
        if w in pretrainVocab:
            key_word2vec[w] = pretrainVec.word_vec(w)
    print(len(key_word2vec))
    return key_word2vec


news = pd.read_csv('/Users/marcowang/Dropbox/Textnews/TextProject/sample_data/news20190406_285.csv',index_col=None)
newsText = list(news['NewsContent'])
flatten = lambda l:[item for sublist in  l for item in sublist]
newsToken = [regexp_tokenize(sent,pattern = '\w+|\$[\d\.]+|\S+') for sent in newsText]
newsToken = flatten(newsToken)
allWords = [w.lower() for w in newsToken]
vocabulary = list(set(allWords))  # news vocabulary 
len(vocabulary) ##   Oh my G~~~~， It is difficult for me to read about techinical news~， so many words ~


##########################################
#Importing wiki vecter####################
##########################################
wiki_en = KeyedVectors.load_word2vec_format('/Users/marcowang/Downloads/text_project/data/word2vec_pretrain_data/wiki-news-300d-1M.vec')
print('=' * 80)
print('Loading Success ！！')
print('=' * 80)

vocab_wiki = list(wiki_en.vocab.keys())
len(vocab_wiki) ### pretrain vocab 


##########################################
#getting  the  vecters####################
##########################################
news2Vec = news2vec(vocabulary,wiki_en,vocab_wiki)

##########################################
#Save the result      ####################
##########################################
df =  pd.DataFrame(news2Vec)

#df.to_csv('newsVectors_9223.csv') 


