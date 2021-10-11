# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""
import sys
sys.path.append('submodule/')

import numpy as np
import tensorflow as tf

import common_attention as ca
import common_layers as cl


attn_dropout = 0.1
residual_dropout=0.1
hidden_size = 300
value_depth = 300
num_heads = 2

def atten_layer(x,mode):
    x = tf.squeeze(x)
    def residual_fn(x, y):
        return cl.layer_norm(x + tf.nn.dropout(y, 1.0 - residual_dropout if mode == 'TRAIN' else 1))
    
    padding = ca.embedding_to_padding(x)
    self_attention_bias = ca.attention_bias_ignore_padding(padding)
    
    print('hfdsjkb/n',x)
    at_X = residual_fn(
    x,
    ca.multihead_attention(
            query_antecedent = x,
            memory_antecedent = None, 
            bias = self_attention_bias,
            total_key_depth = hidden_size,
            total_value_depth = value_depth,
            output_depth = hidden_size,
            num_heads = num_heads,
            dropout_rate = attn_dropout if mode == 'TRAIN' else 0,
            summaries = False,
            image_shapes = None,
            name = None
        ))
    at_X = tf.expand_dims(at_X,3)
    print('at_X',at_X)
  
    
    
    return at_X
def residual_fn(x, y,mode):
        return cl.layer_norm(x + tf.nn.dropout(y, 1.0 - residual_dropout if mode == 'TRAIN' else 1))    