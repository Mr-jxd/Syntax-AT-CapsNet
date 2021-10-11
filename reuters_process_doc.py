"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""
import numpy as np
import h5py
import re
import operator

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

#print(cachedStopWords)

def load_word_vector(fname, vocab):
    model = {}
    with open(fname) as fin:
        for line_no, line in enumerate(fin):
            try:
                parts = line.strip().split(' ')
                word, weights = parts[0], parts[1:]
                if word in vocab:                     
                    model[word] = np.array(weights,dtype=np.float32)
            except:
                pass
    return model

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        #print(header)
        vocab_size, layer1_size = map(int, header.split())
        #print(vocab_size)
        binary_len = np.dtype('float32').itemsize * layer1_size
        #print(binary_len)
        for line in range(vocab_size):
            #print (line)
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab: 
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')  
            else:
               f.read(binary_len)
            #print(word_vecs)
    return word_vecs
#句子变成词
def line_to_words(line):
    words = map(lambda word: word.lower(), word_tokenize(line))
    #print("words",words)    
    tokens = words
    p = re.compile('[a-zA-Z]+')
    return list(filter(lambda token: p.match(token), tokens))        

def get_vocab(dataset):
    max_sent_len = 0
    word_to_idx = {}
    idx = 1
    for line in dataset:    
        words = line_to_words(line)
        #print("words",words)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    return max_sent_len, word_to_idx




def load_data(dataset_name,padding=0, sent_len=300, w2i=None):
    
    """
        threshold = 0  all labels in test data
        threshold = 1  only multilabels in test data
    """
    
    threshold = 1 
    
    train_docs, train_cats, test_docs, test_cats = [], [], [], []
    
    popular_topics = set(['earn','acq','money-fx','grain','crude','trade','interest','ship','wheat','corn'])
    
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            if set(reuters.categories(doc_id)).issubset(popular_topics):
                train_docs.append(reuters.raw(doc_id))
                train_cats.append([cat for cat in reuters.categories(doc_id)])
#            train_cats.append(
#                [cats.index(cat) for cat in reuters.categories(doc_id)])
        else:
            if set(reuters.categories(doc_id)).issubset(popular_topics):
                test_docs.append(reuters.raw(doc_id))
                test_cats.append([cat for cat in reuters.categories(doc_id)])
    
    dataset = train_docs + test_docs
    #print(dataset[0])
    max_sent_len, word_to_idx = get_vocab(dataset)
    print(max_sent_len)
    #print(word_to_idx)
    if sent_len > 0:
        max_sent_len = sent_len      
    if w2i is not None:
        word_to_idx = w2i    

        
    s=open(dataset_name+'train_doc_sig','a')
    train, train_label, test, test_label = [], [], [], []    
    for i, line in enumerate(train_docs):
        #print(line)
        words = line_to_words(line)
        
        

        # length=str(len(words))
        #s.writelines(length)
        #s.writelines('\n')
        #print(words)
        #break
        y = train_cats[i]
        #print(y)
        if len(y) > 1: # The examples which contain at least 1 label would be assigned to test data.
            test_docs.append(line)
            test_cats.append(y)
            continue
        
        y = y[0]
        
        train_label.append(y)
        for word in words:
            s.writelines(word)
            s.writelines(' ')
        s.writelines('\n')
    #print('train_docs',len(train_docs))
    single_label = ['-1'] + list(set(train_label))
    num_classes = len(single_label)
    print('test_docs',len(test_docs))
    p=open(dataset_name +'test_doc_sig','a')          
    for i, line in enumerate(test_docs):
        words = line_to_words(line)
        y = test_cats[i]    
        # sent = [word_to_idx[word] for word in words if word in word_to_idx]
        # if len(sent) > max_sent_len:
        #     sent = sent[:max_sent_len]
        # else:    
        #     sent.extend([0] * (max_sent_len + padding - len(sent)))
        if len(y) == threshold and set(y).issubset(single_label):
            test.append(line)
            for word in words:            
                p.writelines(word)
                p.writelines(' ')
            p.writelines('\n')
            # one_hot_y = np.zeros([num_classes],dtype=np.int32)
            # for yi in y:
            #     one_hot_y[single_label.index(yi)]=1
            #     #print(single_label.index(yi))
            # test_label.append(one_hot_y)        
        # words = line_to_words(line)
        # for word in words:
        # p.writelines(line)
        # p.writelines(' ')
        
        # p.writelines('\n')

#dataset = 'reuters_multilabel_all'
dataset = 'reuters_all_'
load_data(dataset,padding=0, sent_len=300, w2i=None)
#print(test_label)
#print(train[0])

# print(type(train_[0]))
# print(train_[:10])
# # train_adj=adj_get(train_)
# test_adj=adj_get(test_)

# print('train_adj',train_adj[0][:4])
# print('test_adj',test_adj[0][:4])
# def adj_put(train_adj,test_adj):
#     filename = dataset + '_adj.hdf5'
#     with h5py.File(filename, "w") as f:
#         f["train_adj"] = train_adj
#         f['test_adj'] = test_adj
