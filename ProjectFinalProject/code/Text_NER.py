####################Import Packages##############################
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords # import packge
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from spacy import displacy
from collections import Counter
import en_core_web_sm

#####################Read file and Preprocessing##################

news = pd.read_csv('news20190406_285.csv')
newsdocument = list(news['NewsContent']) # a list of news with len = 285

def docu_preprocess(News): # newsdocument[i], one single news
    '''input: a news
       output: a list of tuples containing the individual words in the sentence and their associated part-of-speech'''
    #newsdocument = str(';'.join(News))                     # join all news into a string
    newsToken = nltk.word_tokenize(News)                    # Tokenize the news
    pos_tag = nltk.pos_tag(newsToken)                       #return pos_tag of the news
    return pos_tag

newsToken_POS = [docu_preprocess(i) for i in newsdocument] # list of news token using docu_preprocess

##################### ______NLTK _________#########################


# Use NLTK Package to generate NER and POS_tag in each news
iob_tagList = []
ne_treeList = []
for i in range(len(newsToken_POS)):
    pattern = 'NP: {<DT>?<JJ>*<NN>}'                            # chunk pattern, can be modified latter
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(newsToken_POS[i])
    iob_tagged = tree2conlltags(cs) # return a list of tuples includes text,POS_tags, and classification of NER
    iob_tagList.append(iob_tagged)
    ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(newsdocument[i]))) # use function nltk.ne_clunk to reconginize named entity using a classifier
    ne_treeList.append(ne_tree)

#print(iob_tagList) # list of tuples, contain text,POS_tags, and classification of NER
#print(ne_treeList)

print('iob_tagList len is ', len(iob_tagList))
print('ne_treeList len is ', len(ne_treeList))

#####################_______SpaCy_________#########################
# use SpaCy Package to generate NER and POS_tag in each news
nlp = en_core_web_sm.load()
for i in range(len(newsdocument)):
    doc = nlp(str(newsdocument[i]))
    SpaCy_POS =[(x.orth_,x.pos_,x.tag_,x.dep_) for x in doc] # retuen Verbatim text content, POS, Tag,
    SpaCy_NER = [(x.text, x.label_) for x in doc.ents] # return text, NER

#print(SpaCy_POS)
#print(SpaCy_NER)

print('SpaCy_POS len is ', len(iob_tagList))
print('SpaCy_NER len is ', len(ne_treeList))



















