# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""


from numpy import *


import nltk
import stanfordnlp
from collections import defaultdict
#stanfordnlp.download('en', force=True)
nlp = stanfordnlp.Pipeline()

#input sentences list - output adjs
def adj_get(sentences):        
    k=0
    nodes_len=0
    tree_num=1
    adjs=[]
    for sentence in sentences:
        adj = mat(zeros((201,201)))
        #print(sentence)
        # sentence=array(sentence)
        # print(sentence)
        # if sentence is None or sentence.size <= 1:
        if sentence is None or len(sentence) <= 1:
            #print(1)
            continue
        nodes = []
        #print (sentence)
        doc = nlp(sentence)
        #print(doc.sentences)
        for tree in doc.sentences:
            # s.writelines('tree')
            # s.writelines('\n')
            #break
            #print('tree',tree)
            for edge in tree.dependencies:
                #print('edge',edge)
                
                if tree_num >1:
                    nodes.append((nodes_len,int(edge[0].index)+nodes_len,edge[0].text,edge[0].text))
                    tree_num=1
                nodes.append((int(edge[0].index)+nodes_len,int(edge[2].index)+nodes_len,edge[0].text,edge[2].text))
                # print('edge0',edge[0])
                # print('edge1',edge[1])
                # print('edge2',edge[2])
                # print(nodes)   
            nodes_len=len(nodes)
            #print('nodes_len',nodes_len) 
            tree_num=tree_num+1
        tree_num=1
        nodes_len=0
        #print (nodes)
        
        for node in nodes:
            #print(len(node))
            if int(node[0])<201 and int(node[1])<201:
                adj[int(node[0]),int(node[1])]=1
        
        # #print(adj[:4])
        # for i in range(201):
            # for j in range(201):
                # if i==j:
                    # adj[i,j]=1
#        adj=delete(adj,0,axis=1)
#        adj=delete(adj,0,axis=0)
        adjs.append(adj)	
        print(k)
        k=k+1
        # if k==5:
            # break
    return adjs
        # break


# with open(path_name, 'r') as f:
#     sentences = f.readlines()
#     # if len(sentences) <= 0:
#     #     sentences = f.readlines()
#     #print(type(sentence))
# print(type(sentences))
# print(sentences[:10])   
# adjs=adj_get(sentences)
# print(adjs[0][:4])

# array_1 = random.rand(4,4)
# print(array_1)
# print(type(array_1))
# nodes = [[1, 2],[1,3],[2,3]]
# print (nodes)
# adj = mat(zeros((200,200)))
# print(adj)
# print(type(adj))

# for i in nodes:
#     adj[int(i[0])-1,int(i[1])-1]=1
# print(adj)
    