# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""

from adj_get import adj_get
import h5py




def get_doc(path_name):
    f=open(path_name, 'r')
    sentences= f.readlines()
    return sentences
def get_adj(sentences):
    adj=adj_get(sentences)
    return adj
def adj_put(dataset,train_adj,test_adj):
    filename = dataset + '_adj.hdf5'
    with h5py.File(filename, "w") as f:
        f["train_adj"] = train_adj
        f['test_adj'] = test_adj
def doc_to_adj(dataset):
    train_name=dataset+'train_doc'
    test_name=dataset+'test_doc'
    train_doc=get_doc(train_name)
    test_doc=get_doc(test_name)      
    train_adj=get_adj(train_doc)
    test_adj=get_adj(test_doc)
    print('train_adj',type(train_adj))
    print('test_adj',type(test_adj))
    adj_put(dataset,train_adj,test_adj)
dataset = 'reuters_all'
doc_to_adj(dataset)

      
        