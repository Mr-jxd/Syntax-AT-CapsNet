"""
Created on Sun Jan  5 21:11:08 2021

@author: Mr.JXD
"""

from __future__ import division, print_function, unicode_literals
import argparse
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from loss import spread_loss, cross_entropy, margin_loss
from network_G import baseline_model_kimcnn, baseline_model_cnn, capsule_model_A, capsule_model_B, capsule_model_at,capsule_model_at_sq,baseline_model_kimcnn_G
from sklearn.utils import shuffle
from keras import utils
tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--embedding_type', type=str, default='static',
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--dataset', type=str, default='reuters_multilabel_all',
                    help='Options: reuters_multilabel_dataset,reuters_multilabel_all')

parser.add_argument('--loss_type', type=str, default='margin_loss',
                    help='margin_loss, spread_loss, cross_entropy')

parser.add_argument('--model_type', type=str, default='capsule-A',
                    help='capsule-A,capsule-at,capsule-at-sq,CNN,KIMCNN,KIMCNN-sq')

parser.add_argument('--has_test', type=int, default=1, help='If data has test, we use it. Otherwise, we use CV on folds')    
parser.add_argument('--has_dev', type=int, default=1, help='If data has dev, we use it, otherwise we split from train')    

parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=25, help='Batch size for training')

parser.add_argument('--use_orphan', type=bool, default='True', help='Add orphan capsule or not')
parser.add_argument('--use_leaky', type=bool, default='False', help='Use leaky-softmax or not')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate for training')#CNN 0.0005 
parser.add_argument('--margin', type=float, default=0.1, help='the initial value for spread loss')

import json
args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))


def load_data(dataset):
    train, train_label = [],[]
    dev, dev_label = [],[]
    test, test_label = [],[]
    
    f = h5py.File(dataset+'.hdf5', 'r') 
    print('loading data...')    
    print(dataset)
    print("Keys: %s" % f.keys())
  
    w2v = list(f['w2v'])
    #print(len(w2v[0]))
    train = list(f['train'])
    print('len_train',len(train))
    train_label = list(f['train_label'])
    if args.use_orphan:
        args.num_classes = max(train_label) + 1
        args.num_classes = 9       
    if len(list(f['test'])) == 0:
        args.has_test = 0
    else:
        args.has_test = 1
        test = list(f['test'])
        test_label = list(f['test_label'])
    
    for i, v in enumerate(train):
        if np.sum(v) == 0:        
            del(train[i])     
            del(train_label[i])
    
    for i, v in enumerate(test):
        if np.sum(v) == 0:
            del(test[i])
            del(test_label[i])
    

    train, dev, train_label, dev_label = train_test_split(train, train_label, test_size=474, random_state=0)
    return train, train_label, test, test_label, dev, dev_label, w2v
#daoru adj
def load_adj(dataset):
    train_adj,test_adj= [],[]
    f = h5py.File(dataset+'_adj'+'.hdf5', 'r') 
    print('loading data...')    
    print("Keys: %s" % f.keys())  
    train_adj= list(f['train_adj'])
    test_adj= list(f['test_adj'])
    train_adj=train_adj[:5474]
    train_adj, dev_adj=train_test_split(train_adj,test_size=474, random_state=0)
    return train_adj, dev_adj,test_adj
    # print(len(test_adj))
    # print(type(train_adj[0]))
    # print(train_adj[0].shape)
    # print(test_adj[0][:4])
    
class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, dataset,label,adj, batch_size,input_size, is_shuffle=True):
      self._dataset = dataset
      self._label = label
      self._adj = adj
      self._batch_size = batch_size    
      self._cursor = 0      
      self._input_size = input_size      
      
      if is_shuffle:
          index = np.arange(len(self._dataset))
          np.random.shuffle(index)
          self._dataset = np.array(self._dataset)[index]
          self._label = np.array(self._label)[index]
          self._adj = np.array(self._adj)[index]
      else:
          self._dataset = np.array(self._dataset)
          self._label = np.array(self._label)
          self._adj = np.array(self._adj)
    def next(self):
      if self._cursor + self._batch_size > len(self._dataset):
          self._cursor = 0
      """Generate a single batch from the current cursor position in the data."""      
      batch_x = self._dataset[self._cursor : self._cursor + self._batch_size,:]
      batch_y = self._label[self._cursor : self._cursor + self._batch_size]
      batch_adj = self._adj[self._cursor : self._cursor + self._batch_size]
      self._cursor += self._batch_size
      return batch_x, batch_y,batch_adj

train, train_label, test, test_label, dev, dev_label, w2v= load_data(args.dataset)    
train_adj, dev_adj,test_adj=load_adj(args.dataset)

# print('len_train',len(train))
# print('len_train',len(train_label))
# print('len_train',len(test))
# print('len_train',len(test_label))
# print('len_train',len(dev))
# print('len_train',len(dev_label))
# print('len_train',len(train_adj))
# print('len_train',len(test_adj))
# print('len_train',len(dev_adj))

train=train[:5000]
train_label=train_label[:5000]
test=test[:2500]
test_label=test_label[:2500]
dev=dev[:474]
dev_label=dev_label[:474]
train_adj=train_adj[:5000]
dev_adj=dev_adj[:474]
test_adj=test_adj[:2500]

print('len_train',len(train))
print('len_train',len(train_label))
print('len_train',len(test))
print('len_train',len(test_label))
print('len_train',len(dev))
print('len_train',len(dev_label))
print('len_train',len(train_adj))
print('len_train',len(test_adj))
print('len_train',len(dev_adj))
print('len_train',train_adj[0])
print('len_train',len(train))
# print(train_label[0:10])
#print("dfd",utils.to_categorical(train_label, args.num_classes)[0:10]  )
args.vocab_size = len(w2v)
args.vec_size = w2v[0].shape[0]
args.max_sent = len(train[0])
print('max sent: ', args.max_sent)
print('vocab size: ', args.vocab_size)
print('vec size: ', args.vec_size)
print('num_classes: ', args.num_classes)

train, train_label, train_adj= shuffle(train, train_label,train_adj)
print('len_train',train_adj[0])
with tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()

label = ['-1', 'earn', 'money-fx', 'trade', 'acq', 'grain', 'interest', 'crude', 'ship']
label = map(str,label)
args.max_sent = 10
threshold = 0.5

X = tf.placeholder(tf.int32, [args.batch_size, args.max_sent], name="input_x")
y = tf.placeholder(tf.int64, [args.batch_size, args.num_classes], name="input_y")
X_adj=tf.placeholder(tf.float32, [args.batch_size, args.max_sent,args.max_sent], name="input_x")
is_training = tf.placeholder_with_default(False, shape=())    
learning_rate = tf.placeholder(dtype='float32') 
margin = tf.placeholder(shape=(),dtype='float32') 

l2_loss = tf.constant(0.0)

w2v = np.array(w2v,dtype=np.float32)
if args.embedding_type == 'rand':
    W1 = tf.Variable(tf.random_uniform([args.vocab_size, args.vec_size], -0.25, 0.25),name="Wemb")
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[...,tf.newaxis] 
if args.embedding_type == 'static':
    W1 = tf.Variable(w2v, trainable = False)
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[...,tf.newaxis] 
if args.embedding_type == 'nonstatic':
    W1 = tf.Variable(w2v, trainable = True)
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[...,tf.newaxis] 
if args.embedding_type == 'multi-channel':
    W1 = tf.Variable(w2v, trainable = True)
    W2 = tf.Variable(w2v, trainable = False)
    X_1 = tf.nn.embedding_lookup(W1, X)
    X_2 = tf.nn.embedding_lookup(W2, X) 
    X_1 = X_1[...,tf.newaxis]
    X_2 = X_2[...,tf.newaxis]
    X_embedding = tf.concat([X_1,X_2],axis=-1)

tf.logging.info("input dimension:{}".format(X_embedding.get_shape()))
#---
if args.model_type == 'capsule-A':    
    poses, activations = capsule_model_A(X_embedding, args.num_classes)    
if args.model_type == 'capsule-B':    
    poses, activations = capsule_model_B(X_embedding, args.num_classes)
if args.model_type == 'capsule-at':    
    poses, activations = capsule_model_at(X_embedding, args.num_classes)
if args.model_type == 'capsule-at-sq':    
    poses, activations = capsule_model_at_sq(X_embedding, args.num_classes,X_adj)    
if args.model_type == 'CNN':    
    poses, activations = baseline_model_cnn(X_embedding, args.num_classes)
if args.model_type == 'KIMCNN':    
    poses, activations = baseline_model_kimcnn(X_embedding, args.max_sent, args.num_classes)   
if args.model_type == 'KIMCNN-sq':    
    poses, activations = baseline_model_kimcnn_G(X_embedding, args.max_sent, args.num_classes,X_adj)    
if args.loss_type == 'spread_loss':
    loss = spread_loss(y, activations, margin)
if args.loss_type == 'margin_loss':    
    loss = margin_loss(y, activations)
if args.loss_type == 'cross_entropy':
    loss = cross_entropy(y, activations)

y_pred = tf.argmax(activations, axis=1, name="y_proba")
#print("y_pred",activations)    
correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)   
training_op = optimizer.minimize(loss, name="training_op")
gradients, variables = zip(*optimizer.compute_gradients(loss))

grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
              for g in gradients if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]
with tf.control_dependencies(grad_check):
    training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)      


sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
# tf.reset_default_graph()

n_iterations_per_epoch = len(train) // args.batch_size
n_iterations_test = len(test) // args.batch_size
n_iterations_dev = len(dev) // args.batch_size    

mr_train = BatchGenerator(train,train_label,train_adj ,args.batch_size, 0)    
mr_dev = BatchGenerator(dev,dev_label,dev_adj, args.batch_size, 0)
mr_test = BatchGenerator(test,test_label, test_adj,args.batch_size, 0, is_shuffle=False)
X_batch, y_batch,X_adj_batch = mr_train.next()
print('len_train',len(X_batch))
print('len_train',len(y_batch))
print('len_train',len(X_adj_batch)) 
best_model = None
best_epoch = 0
best_acc_val = 0.

init = tf.global_variables_initializer()
sess.run(init)     

lr = args.learning_rate
m = args.margin
for epoch in range(args.num_epochs):
    
    for iteration in range(1, n_iterations_per_epoch + 1):                
        X_batch, y_batch,X_adj_batch = mr_train.next()     
        y_batch = utils.to_categorical(y_batch, args.num_classes)        
        _, loss_train, probs, capsule_pose = sess.run(
            [training_op, loss, activations, poses],
            feed_dict={X: X_batch[:,:args.max_sent],
                        y: y_batch,
                        X_adj:X_adj_batch,
                        is_training: True,
                        learning_rate:lr,
                        margin:m})        
        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                  iteration, n_iterations_per_epoch,
                  iteration * 100 / n_iterations_per_epoch,
                  loss_train),
              end="")
        #print(probs[0])                        
    loss_vals, acc_vals = [], []
    for iteration in range(1, n_iterations_dev + 1):
        X_batch, y_batch,X_adj_batch  = mr_dev.next()            
        #y_batch = utils.to_categorical(y_batch, args.num_classes)
        loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch[:,:args.max_sent],
                            y: y_batch,
                            X_adj:X_adj_batch,
                            is_training: False,
                            margin:m})
        loss_vals.append(loss_val)
        acc_vals.append(acc_val)
    print("loss_vals",loss_vals)
    print("acc_vals",acc_vals)
    loss_val, acc_val = np.mean(loss_vals), np.mean(acc_vals)    
    print("\rEpoch: {}  Val accuracy: {:.1f}%  Loss: {:.4f}".format(
        epoch + 1, acc_val * 100, loss_val))
               
    preds_list, y_list ,acc_vals= [], [],[]

    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch,X_adj_batch = mr_test.next()             
        probs = sess.run([activations],
                feed_dict={X:X_batch[:,:args.max_sent],
                           X_adj:X_adj_batch,
                            is_training: False})
        print('probs':probs)
        print('probs[0]':probs[0])
        preds_list = preds_list + probs[0].tolist()
        #print(preds_list)
        y_list = y_list + y_batch.tolist()  
    y_list = np.array(y_list)
    preds_probs = np.array(preds_list)                
    preds_probs[np.where( preds_probs >= threshold )] = 1.0
    preds_probs[np.where( preds_probs < threshold )] = 0.0 
    
    [precision, recall, F1, support] = \
        precision_recall_fscore_support(y_list, preds_probs, average='micro')
    acc = accuracy_score(y_list, preds_probs)

    print ('\rER: %.3f' % acc, 'Precision: %.3f' % precision, 'Recall: %.3f' % recall, 'F1: %.3f' % F1)  
    if args.model_type == 'CNN' or args.model_type == 'KIMCNN':
        lr = max(1e-6, lr * 0.8)
    if args.loss_type == 'margin_loss':    
        m = min(0.9, m + 0.1)
