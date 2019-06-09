
#@author Marco Wang
#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import pandas as pd
from nltk.tokenize import regexp_tokenize
from gensim.models import KeyedVectors
from gensim.models import word2vec
from gensim.scripts import glove2word2vec
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(224)

##########################################
#Importing wiki vector####################
##########################################
## Mac
# = KeyedVectors.load_word2vec_format('/Users/marcowang/Downloads/text_project/data/word2vec_pretrain_data/wiki-news-300d-1M.vec')

# Ubantu
wiki_en = KeyedVectors.load_word2vec_format('/home/marco/Downloads/wiki-news-300d-1M.vec')

print('=' * 80)
print('Loading Success ！！')
print('=' * 80)

vocab_wiki = list(wiki_en.vocab.keys())
print(len(vocab_wiki) ### pretrain vocab)

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin
google = load_word2vec() ## another pretrain vectors


# For GPU use
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

# Batch
def getBatch(batch_size, train_data):
        random.shuffle(train_data)
        sindex = 0
        eindex = batch_size
        while eindex < len(train_data):
            batch = train_data[sindex: eindex]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch

        if eindex >= len(train_data):
            batch = train_data[sindex:]
            yield batch


#Padding
def pad_to_batch(batch,max_x = 1024):
    x,y = zip(*batch)
    
    x_p = []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i], Variable(LongTensor([word2index['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
    return torch.cat(x_p), torch.cat(y).view(-1)

#seq text transform
def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

#loading data
#Mac
#news = pd.read_csv('/Users/marcowang/Dropbox/Textnews/TextProject/sample_data/news20190406_285.csv',index_col=None)

#Ubantu
news = pd.read_csv('/home/marco/Dropbox/Textnews/TextProject/sample_data/news20190406_285.csv',index_col=None)

newsText = list(news['NewsContent'])
flatten = lambda l:[item for sublist in  l for item in sublist]
newsToken = [regexp_tokenize(sent,pattern = '\w+|\$[\d\.]+|\S+') for sent in newsText] #####?
new_token = [x for x in newsToken if len(x)<=1000]
flatToken = flatten(new_token)
allWords = [w.lower() for w in flatToken]
vocabulary = list(set(allWords))  # news vocabulary 
print(len(vocabulary))


#Mac
#data = pd.read_csv('/Users/marcowang/Dropbox/Textnews/TextProject/sample_data/tagged_data/finalInput.csv',index_col=0)

#Ubantu
data = pd.read_csv('/home/marco/Dropbox/Textnews/TextProject/sample_data/tagged_data/finalInput.csv',index_col=0)


y_com = data.iloc[:,4:9].values # company info
y_sent = data.iloc[:,9:].values # 

# negative impact label as '2'
for i in range(y_sent.shape[0]):
    for j in range(y_sent.shape[1]):
        if y_sent[i][j] == -1:
            y_sent[i][j] = 2
        
#word 2 index in corpus
word2index={'<PAD>': 0, '<UNK>': 1} # pad means padding !

for vo in vocabulary:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
        
index2word = {v:k for k, v in word2index.items()}

#data preparation
X_p, y_p = [],[]

for x,y in zip(newsToken,y_sent):
    if len(x) <=1000:
        X_p.append(prepare_sequence(x, word2index).view(1, -1))
        y_p.append(Variable(LongTensor(y)).view(1, -1))
        
    
data_p = list(zip(X_p,y_p))
random.shuffle(data_p)

train_data = data_p[: int(len(data_p) * 0.8)]
test_data = data_p[int(len(data_p) * 0.8):]

#traning data for a company
y_ap = [x[1][0][0].view(1, -1) for x in train_data ]
y_bm = [x[1][0][1].view(1, -1) for x in train_data ]
y_go = [x[1][0][2].view(1, -1) for x in train_data ]
y_am = [x[1][0][3].view(1, -1) for x in train_data ]
y_ms = [x[1][0][4].view(1, -1) for x in train_data ]
train_x = [x[0] for x in train_data ]
train_ap = list(zip(train_x,y_ap))
train_bm = list(zip(train_x,y_bm))
train_go = list(zip(train_x,y_go))
train_am = list(zip(train_x,y_am))
train_ms = list(zip(train_x,y_ms))

#pretrain vector
pretrained = []

for key in word2index.keys():
    try:
        pretrained.append(wiki_en[word2index[key]])
    except:
        pretrained.append(np.random.randn(300))
        
pretrained_vectors = np.vstack(pretrained)



##Model Structure 
class Sent(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(Sent,self).__init__()
        self.embedding_w = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(     # 
        input_size = 300,      # vecter dimension
        hidden_size=64,     # rnn hidden unit
        num_layers=1,       # 
        batch_first=True,   # input & output 
        )
        self.out = nn.Linear(64, 3) #  3 class
        
    
    
    def init_weights(self, pretrained_word_vectors, is_static=True):
        self.embedding_w.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding_w.weight.requires_grad = False


    def forward(self, inputs, is_training=False):
        inputs = self.embedding_w(inputs) # (B,1,T,D)
        r_out, (h_n, h_c) = self.rnn(inputs, None)
        out = self.out(r_out[:, -1, :])
        return out


#Apple
EPOCH = 100
BATCH_SIZE = 20
LR = 0.001
TIME_STEP = 32          # rnn time step / image height
INPUT_SIZE = 32

apmodel = Sent(len(word2index), 300)
apmodel.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(apmodel.parameters(), lr=LR)

for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_ap)):

        inputs,targets = pad_to_batch(batch)
        #print(targets.size())          
        apmodel.zero_grad()
        preds = apmodel(inputs,True)
        #print(preds.size())
        loss = loss_function(preds, targets)
        losses.append(loss.data.mean())
        loss.backward()
        
        
        optimizer.step()
        acc = 0
        if i % 100 == 0:
            for test in test_data:
                oup = apmodel(test[0])
                a = (torch.max(oup,1)[1].item())
                b = (test[1][0][0].item())
                if a == b:
                    acc +=1
            accuracy = acc/len(test_data)
            print('Epoch: ', str(epoch)+'/'+str(EPOCH), '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


#Google
EPOCH = 100
BATCH_SIZE = 50
LR = 0.001
TIME_STEP = 1024         # 
INPUT_SIZE = 1

gomodel = Sent(len(word2index), 300)
gomodel.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(gomodel.parameters(), lr=LR)

for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_go)):

        inputs,targets = pad_to_batch(batch)
        #print(targets.size())          
        gomodel.zero_grad()
        preds = gomodel(inputs,True)
        #print(preds.size())
        loss = loss_function(preds, targets)
        losses.append(loss.data.mean())
        loss.backward()
        
        #for param in model.parameters():
        #    param.grad.data.clamp_(-3, 3)
        
        optimizer.step()
        acc = 0
        if i % 100 == 0:
            for test in test_data:
                oup = gomodel(test[0])
                a = (torch.max(oup,1)[1].item())
                b = (test[1][0][0].item())
                if a == b:
                    acc +=1
            accuracy = acc/len(test_data)
            print('Epoch: ', str(epoch)+'/'+str(EPOCH), '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

#Amazon

EPOCH = 100
BATCH_SIZE = 50
LR = 0.001
TIME_STEP = 1024         # 
INPUT_SIZE = 1

ammodel = Sent(len(word2index), 300)
ammodel.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(ammodel.parameters(), lr=LR)

for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_am)):

        inputs,targets = pad_to_batch(batch)
        #print(targets.size())          
        ammodel.zero_grad()
        preds = ammodel(inputs,True)
        #print(preds.size())
        loss = loss_function(preds, targets)
        losses.append(loss.data.mean())
        loss.backward()
        
        #for param in model.parameters():
        #    param.grad.data.clamp_(-3, 3)
        
        optimizer.step()
        acc = 0
        if i % 100 == 0:
            for test in test_data:
                oup = ammodel(test[0])
                a = (torch.max(oup,1)[1].item())
                b = (test[1][0][0].item())
                if a == b:
                    acc +=1
            accuracy = acc/len(test_data)
            print('Epoch: ', str(epoch)+'/'+str(EPOCH), '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


#Microsoft
EPOCH = 100
BATCH_SIZE = 50
LR = 0.001
TIME_STEP = 1024         # 
INPUT_SIZE = 1

msmodel = Sent(len(word2index), 300)
msmodel.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(msmodel.parameters(), lr=LR)

for epoch in range(EPOCH):
    losses = []
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_ms)):

        inputs,targets = pad_to_batch(batch)
        #print(targets.size())          
        msmodel.zero_grad()
        preds = msmodel(inputs,True)
        #print(preds.size())
        loss = loss_function(preds, targets)
        losses.append(loss.data.mean())
        loss.backward()
        
        #for param in model.parameters():
        #    param.grad.data.clamp_(-3, 3)
        
        optimizer.step()
        acc = 0
        if i % 100 == 0:
            for test in test_data:
                oup = msmodel(test[0])
                a = (torch.max(oup,1)[1].item())
                b = (test[1][0][0].item())
                if a == b:
                    acc +=1
            accuracy = acc/len(test_data)
            print('Epoch: ', str(epoch)+'/'+str(EPOCH), '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score


dictCompany = {apmodel:0,bmmodel:1,gomodel:2,ammodel:3,msmodel:4}

def getEvaluation(model_name,testData):
    y_pred = []
    y_test = []
    n = dictCompany[model_name]
    for test in testData:
        oup = model_name(test[0])
        y_p = (torch.max(oup,1)[1].item())
        y_pred.append(y_p)
        y_t = (test[1][0][n].item())
        y_test.append(y_t)
    #fpr, tpr,threshold = roc_curve(y_test)
    print(classification_report(y_test,y_pred))
    print(pd.DataFrame(data = confusion_matrix(y_test,y_pred),columns = ['Pre_Flat','Pre_Up','Pre_Down'],
          index = ['Flat','Up','Down']))

    

#apple
print(getEvaluation(apmodel,test_data))

#ibm
print(getEvaluation(bmmodel,test_data))

#google
print(getEvaluation(gomodel,test_data))

#amazon
print(getEvaluation(ammodel,test_data))

#microsoft
print(getEvaluation(msmodel,test_data))
\

#for unseen model
df_test = pd.read_csv('/home/marco/Dropbox/Textnews/TextProject/code/predictResult.csv')
final_test = list(df_test['NewsContent'])

finatestnewsToken = [regexp_tokenize(sent,pattern = '\w+|\$[\d\.]+|\S+') for sent in final_test] #####?
finatestnew_token = [x for x in finatestnewsToken if len(x)<=1000]
finatestflatToken = flatten(finatestnew_token)
finatestallWords = [w.lower() for w in finatestflatToken]
finatestvocabulary = list(set(finatestallWords))  # news vocabulary 
len(finatestvocabulary)

testy_sent = df_test.iloc[:,8:].values
len(testy_sent)

for vo in finatestvocabulary:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
        
index2word = {v:k for k, v in word2index.items()}

X_t, y_t = [],[]

for x,y in zip(finatestnewsToken,testy_sent):
    if len(x) <=1000:
        X_t.append(prepare_sequence(x, word2index).view(1, -1))
        y_t.append(Variable(LongTensor(y)).view(1, -1))
        
    
data_t = list(zip(X_t,y_t))
random.shuffle(data_t)

pretrained = []

for key in word2index.keys():
    try:
        pretrained.append(wiki_en[word2index[key]])
    except:
        pretrained.append(np.random.randn(300))
        
pretrained_vectors = np.vstack(pretrained)



def getPrediction(model_name,testData):
    y_pred = []
    y_test = []
    n = dictCompany[model_name]
    for test in testData:
        oup = model_name(test[0])
        y_p = (torch.max(oup,1)[1].item())
        y_pred.append(y_p)
        y_t = (test[1][0][n].item())
        y_test.append(y_t)
    return y_pred


#apple
apmodel.init_weights(pretrained_vectors)
appleresualt = getPrediction(apmodel,data_t)

#ibm
bmmodel.init_weights(pretrained_vectors)
ibmresualt = getPrediction(bmmodel,data_t)

#google
gomodel.init_weights(pretrained_vectors)
googleresualt = getPrediction(gomodel,data_t)

#Amazon
ammodel.init_weights(pretrained_vectors)
amazonresualt = getPrediction(ammodel,data_t)

#Microsoft
ammodel.init_weights(pretrained_vectors)
amazonresualt = getPrediction(ammodel,data_t)

companyList = [True if len(x)<1000 else False for x in finatestnewsToken ]

answertabel = df_test.iloc[companyList,3:]
answertable = answertabel.reset_index()
answertable = answertable.drop('index',axis =1)
answertable = answertable.drop('index.1',axis =1)
answertable['APPL_senti'] = appleresualt
answertable['IBM_senti'] = ibmresualt
answertable['GOOGL_senti'] = googleresualt
answertable['AMZN_senti'] = amazonresualt
answertable['MSFT_senti'] = msresualt
answertable = answertable.replace(2,-1)
answertable.to_csv('toMfinal predict.csv')



import seaborn as sns
sns.set_style()
objects = ('Apple', 'IBM', 'Google', 'Amazon', 'Microsoft')
y_pos = np.arange(len(objects))

plt.figure(figsize=(8,6),dpi=300)
plt.bar(y_pos, f1, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Micro F1 Score')
plt.title('F1 SCORE of Model Performance')
 
plt.show()