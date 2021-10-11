"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""
import tensorflow as tf
import h5py
import numpy as np
# from utils import _get_weights_wrapper
# def load_adj(dataset):
#     train_adj,test_adj= [],[]
#     f = h5py.File('reuters_all_adj'+'.hdf5', 'r') 
#     print('loading data...')    
#     print("Keys: %s" % f.keys())  
#     train_adj= list(f['train_adj'])
#     test_adj= list(f['test_adj'])

#     return train_adj,test_adj
#     # print(len(test_adj))
#     # print(type(train_adj[0]))
#     # print(train_adj[0].shape)
#     # print(test_adj[0][:4])

# train_adj,test_adj=load_adj('reuters_all_adj')
# print(type(train_adj[0][0][0]))
# print(len(train_adj[0]))

# A=np.ones((50,201,300),dtype=np.float32)
# train_adj=np.array(train_adj).astype(np.float32)
# print(train_adj.shape)
# kernel = _get_weights_wrapper(
#       name='weights', shape=shape, weights_decay_factor=0.0, #initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
#     )

def GCN(A,X,W1,W2):
    # W1=tf.get_variable('',shape)
    # (tf.random_normal(shape))
    # W1=_get_weights_wrapper(name='111', shape=shape)
    # W2=_get_weights_wrapper(name='222', shape=shape)
    # W2=tf.Variable(tf.random_normal(shape))
    # W2=tf.get_variable('',shape)
    a=tf.nn.relu(tf.matmul(tf.matmul(A, X), W1))
    b=tf.nn.softmax(tf.matmul(tf.matmul(A, a), W2))
    # a=tf.nn.relu(tf.matmul(A, X))
    # b=tf.nn.softmax(tf.matmul(A, a))
    return b

#不训练G
# def GCN(A,X,shape=[300,300]):
#     # W1=tf.get_variable('',shape)
#     # (tf.random_normal(shape))
#     # W1=_get_weights_wrapper(name='111', shape=shape)
#     # W2=_get_weights_wrapper(name='222', shape=shape)
#     # W2=tf.Variable(tf.random_normal(shape))
#     # W2=tf.get_variable('',shape)
#     # a=tf.nn.relu(tf.matmul(tf.matmul(A, X), W1))
#     # b=tf.nn.softmax(tf.matmul(tf.matmul(A, a), W2))
#     a=tf.nn.relu(tf.matmul(A, X))
#     b=tf.nn.softmax(tf.matmul(A, a))
#     return b


#     return b

# sess = tf.Session()
# gcs=[]
# for i in range(train_adj.shape[0]):
#     gc=GCN(train_adj[i],A[i])
#     gcs.append(gc)
# sess.run(tf.global_variables_initializer())
# # sess.run(W2.initializer())

# g=sess.run(gcs)
# g=np.array(g)
# print(g.shape)