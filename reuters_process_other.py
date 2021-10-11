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

from nltk.corpus import stopwords
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
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
    return list(filter(lambda token: p.match(token) and len(token) >= 3, tokens))        

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



def load_txt(path_name):
    f=open(path_name, 'r')
    sentences= f.readlines()
    print(len(sentences))
    trains,trains_label=[],[]
    for sentence in sentences:
        if sentence is None or len(sentence) <= 1:
            continue
        # print(type(sentence))
        # print(sentence[:1])
        train=sentence[2:-2]
        train_label=sentence[:1]
        # print(train)
        # print(train_label)
        trains.append(train)   
        trains_label.append(train_label)  
    # print(trains[0:10])
    # print(trains_label[0:10])
    # print(trains[0])
    # a=line_to_words(trains[0])
    # print(a)
    train_docs,test_docs, train_cats, test_cats= train_test_split(trains, trains_label, test_size=0.1, random_state=0)
    # print(train_docs[0:10])
    # print(train_cats[0:10])
    # print(test_docs[0:10])
    # print(test_cats[0:10])

    return train_docs, train_cats, test_docs, test_cats
# load_txt('custrev.all')    


def load_data(path_name,padding=0, sent_len=65, w2i=None):
    
    """
        threshold = 0  all labels in test data
        threshold = 1  only multilabels in test data
    """
    #s=open('save_patht_len.txt','a')
    threshold = 1 
    
    train_docs, train_cats, test_docs, test_cats = [], [], [], []
    
#     popular_topics = set(['earn','acq','money-fx','grain','crude','trade','interest','ship','wheat','corn'])
    

    train_docs, train_cats, test_docs, test_cats = load_txt(path_name)
    dataset = train_docs + test_docs
    #print(dataset[0])
    max_sent_len, word_to_idx = get_vocab(dataset)
    print(max_sent_len)
    #print(word_to_idx)
    if sent_len > 0:
        max_sent_len = sent_len      
    if w2i is not None:
        word_to_idx = w2i    

    train, train_label, test, test_label = [], [], [], []
    print(len(train_docs))    
    for i, line in enumerate(train_docs):
        words = line_to_words(line)
        length=str(len(words))
        y = train_cats[i]
        #print(words)
        #print(y)
        #sharper', 'pictures']
        #1
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        #print(sent)
        #[1, 2]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        train.append(sent)
        train_label.append(y)
    single_label = ['-1'] + list(set(train_label))
    num_classes = len(single_label)
    print(num_classes)
    for i, l in enumerate(train_label):
        train_label[i] = single_label.index(l)
        #print(single_label.index(l))
    for i, line in enumerate(test_docs):
        words = line_to_words(line)
        y = test_cats[i]    
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        test.append(sent)    
        one_hot_y = np.zeros([num_classes],dtype=np.int32)
        for yi in y:
            one_hot_y[single_label.index(yi)]=1
            #print(single_label.index(yi))
        test_label.append(one_hot_y)
            #print(one_hot_y)
            #print(test_label)
    return single_label, word_to_idx, np.array(train), np.array(train_label), np.array(test), np.array(test_label)

    #
path_name='custrev.all'
single_label, word_to_idx, train, train_label, test, test_label = load_data(path_name,padding=0, sent_len=65, w2i=None)
#print(test_label)
print(train[0])
print(single_label)
print ('train size:', train.shape)
print ('test size:', test.shape)
print ('train_lable size:', train_label.shape)
print ('test_lable size:', test_label.shape)

dataset = 'custrev'
with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
    embeddings_f.write("*PADDING* 0\n")
    for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
      embeddings_f.write("%s %d\n" % (word, idx))

w2v_path = 'GoogleNews-vectors-negative300.bin'
#glove_path = 'glove.6B/glove.6B.300d.txt'

w2v = load_bin_vec(w2v_path, word_to_idx)
#print(w2v)

V = len(word_to_idx) + 1
print ('Vocab size:', V) 

def compute_embed(V,w2v):
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec # padding word is positioned at index 0
    return embed

embed_w2v = compute_embed(V, w2v)
#embed_glove = compute_embed(V, glove)

print ('train size:', train.shape)
print ('test size:', test.shape)

filename = dataset + '.hdf5'
with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embed_w2v)
    f['train'] = train
    f['train_label'] = train_label
    f['test'] = test
    f['test_label'] = test_label

