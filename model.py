import os
import time

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
from tensorflow.contrib import legacy_seq2seq

import json
import numpy as np
import char_dataset as cd

from utility_functions import ModelFunctions as mf

class Model():

  #TODO - move this into JSON?
  num_epochs = 5
  epoch_size = 500

  rnn_size = 128
  num_layers = 2
  
  seq_length = int(50)
  batch_size = int(50)
  
  grad_clip = 5
  learning_rate = 0.002
  decay_rate = 0.97

  save_dir = "save"

  def __init__(s, infer=False):
    if infer:
      s.bs = bs = 1
      s.sl = sl = 1
    else:
      s.bs = bs = s.batch_size
      s.sl = sl = s.seq_length
    
    #initialize the dataset
    #TODO- remove this hard coding, shoudl be able to initialze to any dataset
    data = open('./data/shakespeare.txt',  'r').read() #only takes plaintext files

    s.dataset = cd.Dataset(data)
    s.vocab_size = s.dataset.distinct_characters
    #TODO- Migrate to actual LTSM Cell. 
    s.cell = cell = MultiRNNCell([BasicLSTMCell(s.rnn_size)] * s.num_layers)

    s.input_data = tf.placeholder(tf.int32, [bs, sl])
    s.targets = tf.placeholder(tf.int32, [bs, sl])
    s.initial_state = cell.zero_state(bs, tf.float32)

    #TODO- work out why I need this variable scope call
    with tf.variable_scope('rnnlm'):
      softmax_w = tf.get_variable("softmax_w", [s.rnn_size, s.vocab_size])
      softmax_b = tf.get_variable("softmax_b", [s.vocab_size])
      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [s.vocab_size, s.rnn_size])
        inputs = tf.split(tf.nn.embedding_lookup(embedding, s.input_data), sl,1 )
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    
    def loop(prev, _):
      prev = tf.matmul(prev, softmax_w) + softmax_b
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)
    #this also works to set the output
    #outputs, last_state = rnn.rnn(s.cell, inputs, initial_state=s.initial_state)

    outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, s.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
    output = tf.reshape(tf.concat(outputs, 1), [-1, s.rnn_size])
    s.logits = tf.matmul(output, softmax_w) + softmax_b
    s.probs = tf.nn.softmax(s.logits)
    loss = legacy_seq2seq.sequence_loss_by_example([s.logits],
            [tf.reshape(s.targets, [-1])],
            [tf.ones([bs * sl])],
            s.vocab_size)
    s.cost = tf.reduce_sum(loss) / bs/ sl
    s.final_state = last_state
    s.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(s.cost, tvars),
            s.grad_clip)
    optimizer = tf.train.AdamOptimizer(s.lr)
    s.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def sample(s, num=200, prime='The '):
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(tf.all_variables())

      #restore latest save from the save directory
      ckpt = tf.train.get_checkpoint_state(s.save_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("loaded model from : " + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
      
      if s.bs != 1 or s.sl != 1: 
        raise ValueError("can only sample from models with a batch size and sequence length of one" )
      
      prime_keys = s.dataset.char_to_num(list(prime))
       
      state = sess.run( s.cell.zero_state(1, tf.float32))
      #declared here, used below to feed into tensorflow
      inp = np.zeros((1,1)) 
      for c in range(len(prime)-1):
        inp[0, 0] = prime_keys[c]
        feed = {s.input_data: inp, s.initial_state:state}
        [state] = sess.run([s.final_state], feed)

      ret = prime_keys
      char = prime_keys[-1]
      for n in range(num):
        inp[0, 0] = int(char)
        feed = {s.input_data: inp, s.initial_state:state}
        [probs, state] = sess.run([s.probs, s.final_state], feed)
         
        char = mf.weighted_pick(probs[0])
        ret.append(char)
      return  ''.join(s.dataset.num_to_char(ret))


  def train(s):
    #TODO - add in a dependecy to generator functions for input and expected data
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(tf.all_variables())
      #TODO- work out hwo to restore a model from a save
      # restore model (if applicable)
      # saver.restore(sess, ckpt.model_checkpoint_path)
      for e in range(s.num_epochs):
        sess.run(tf.assign(s.lr, s.learning_rate* s.decay_rate*e))
        
        s.dataset.reset()    
        
        state = sess.run( s.initial_state)
        print(state)
        save_dir = os.path.join(s.save_dir, 'model.ckpt')
        saver.save(sess,save_dir, global_step = e) 
        print("saving to " +s.save_dir)
        for n in range(s.epoch_size):
          start = time.time()   
          bs = s.bs
          sl = s.sl
          chars = s.dataset.char_to_num(s.dataset.readn((sl *bs)+ 1))
          inps =np.reshape(np.copy(chars[0:bs*sl]), (bs,sl))
          exps = np.reshape(np.copy(chars[1:(bs*sl)+1]),(bs, sl))
          feed = {s.input_data : inps, s.targets:exps, s.initial_state : state}
          train_loss, state, _ = sess.run([s.cost, s.final_state, s.train_op], feed)
          end = time.time()
          print("batch: {}, epoch: {}, train_loss = {:.3f}, time/batch = {:.3f}" \
                   .format(e * s.epoch_size + n,
                         e, train_loss, end - start))




