"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""

from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from utils import _conv2d_wrapper
from layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
import tensorflow.contrib.slim as slim
from atten_utils import atten_layer,residual_fn
from GCN_get import GCN
#four module

#def compression(self, poses, W):
    #poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
    #activations = torch.sqrt((poses ** 2).sum(2))
    #return poses, activations

def baseline_model_cnn(X, num_classes):
    nets = _conv2d_wrapper(
        X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
        add_bias=False, activation_fn=tf.nn.relu, name='conv1'
        )
    nets = slim.flatten(nets)
    tf.logging.info('flatten shape: {}'.format(nets.get_shape()))
    nets = slim.fully_connected(nets, 128, scope='relu_fc3', activation_fn=tf.nn.relu)
    tf.logging.info('fc shape: {}'.format(nets.get_shape()))
    
    activations = tf.sigmoid(slim.fully_connected(nets, num_classes, scope='final_layer', activation_fn=None))
    tf.logging.info('fc shape: {}'.format(activations.get_shape()))
    return tf.zeros([0]), activations

#
def baseline_model_kimcnn(X, max_sent, num_classes):
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):            
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")            
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = 100 * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.sigmoid(slim.fully_connected(h_pool_flat, num_classes, scope='final_layer', activation_fn=None))
    return tf.zeros([0]), activations

def baseline_model_kimcnn_G(X, max_sent, num_classes,A):
    X1=[]
    X111= tf.squeeze(X)
    W1=tf.get_variable('gcn1',shape=[300,300])
    W2=tf.get_variable('gcn2',shape=[300,300])
    for i in range(25):
        with tf.variable_scope('gcn_'+str(i)):
            X11=GCN(A[i],X111[i],W1,W2)
            X1.append(X11)
    X1= tf.expand_dims(X1,3)
    # X=X1
    X=tf.add(X,X1) 
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):            
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")            
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = 100 * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.sigmoid(slim.fully_connected(h_pool_flat, num_classes, scope='final_layer', activation_fn=None))
    return tf.zeros([0]), activations
#
       
def capsule_model_B(X, num_classes):
    poses_list = []
    for _, ngram in enumerate([3,4,5]):
        with tf.variable_scope('capsule_'+str(ngram)): 
            nets = _conv2d_wrapper(
                X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
            nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
            nets = capsule_flatten(nets)
            poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
            poses_list.append(poses)
    
    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
    activations = K.sqrt(K.sum(K.square(poses), 2))
    return poses, activations

def capsule_model_A(X, num_classes):
    print('X',X)
    with tf.variable_scope('capsule_'+str(3)):   
        nets = _conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_conv_layer(nets, shape=[3, 1, 32, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2') 
    return poses, activations

def capsule_model_at(X, num_classes):
    tf.logging.info('input___: {}'.format(X.get_shape()))
    X=atten_layer(X,'TRAIN')
    #print('X',X)
    tf.logging.info('at_shape___: {}'.format(X.get_shape()))
    with tf.variable_scope('capsule_'+str(3)):   
        nets = _conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        #nets=capsules_sq()
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2') 
    return poses, activations

	
def capsule_model_atB(X, num_classes):
    poses_list = []
    X=atten_layer(X,'TRAIN')
    for _, ngram in enumerate([3,4,5]):
        with tf.variable_scope('capsule_'+str(ngram)): 
            nets = _conv2d_wrapper(
                X, shape=[ngram, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = capsules_init(nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
            nets = capsule_conv_layer(nets, shape=[3, 1, 32, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
            nets = capsule_flatten(nets)
            poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
            poses_list.append(poses)    
    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
    activations = K.sqrt(K.sum(K.square(poses), 2))
    return poses, activations

def capsule_model_A_sq(X, num_classes,A):
    print('X',X)
    X1=[]
    X111= tf.squeeze(X)
    W1=tf.get_variable('gcn1',shape=[300,300])
    W2=tf.get_variable('gcn2',shape=[300,300])
    for i in range(25):
        X11=GCN(A[i],X111[i],W1,W2)
        X1.append(X11)
    X= tf.expand_dims(X1,3)
    X=tf.add(X,X1)

    with tf.variable_scope('capsule_'+str(3)):   
        nets = _conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_conv_layer(nets, shape=[3, 1, 32, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2') 
    return poses, activations	
	
def capsule_model_at_sq(X, num_classes,A):
    X_=atten_layer(X,'TRAIN')
    print('X',X)
    X1=[]
    X111= tf.squeeze(X_)
    W1=tf.get_variable('gcn1',shape=[300,300])
    W2=tf.get_variable('gcn2',shape=[300,300])
    for i in range(25):
        X11=GCN(A[i],X111[i],W1,W2)
        X1.append(X11)
    X1= tf.expand_dims(X1,3)

    X=tf.add(X_,X1)  
    with tf.variable_scope('capsule_'+str(3)):   
        nets = _conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        
        
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                              padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2') 
    return poses, activations



