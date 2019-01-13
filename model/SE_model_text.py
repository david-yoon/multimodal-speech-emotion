#-*- coding: utf-8 -*-

"""
what    : Single Encoder Model for text
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from project_config import *

class SingleEncoderModelText:
    
    def __init__(self, dic_size,
                 use_glove,
                 batch_size,
                 encoder_size,
                 num_layer, lr,
                 hidden_dim,
                 dr):
        
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.num_layers = num_layer
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dr = dr
        
        self.encoder_inputs = []
        self.encoder_seq_length =[]
        self.y_labels =[]

        self.M = None
        self.b = None
        
        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        
        if self.use_glove == 1:
            self.embed_dim = 300
        else:
            self.embed_dim = DIM_WORD_EMBEDDING
        
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    

    def _create_placeholders(self):
        print '[launch-text] placeholders'
        with tf.name_scope('text_placeholder'):
            
            self.encoder_inputs  = tf.placeholder(tf.int32, shape=[self.batch_size, self.encoder_size], name="encoder")  # [batch,time_step]
            self.encoder_seq     = tf.placeholder(tf.int32, shape=[self.batch_size], name="encoder_seq")   # [batch] - valid word step
            self.y_labels        = tf.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")
            self.dr_prob         = tf.placeholder(tf.float32, name="dropout")

             # for using pre-trained embedding
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.dic_size, self.embed_dim], name="embedding_placeholder")

    def _create_embedding(self):
        print '[launch-text] create embedding'
        with tf.name_scope('embed_layer'):
            self.embed_matrix = tf.Variable(tf.random_normal([self.dic_size, self.embed_dim],
                                                            mean=0.0,
                                                            stddev=0.01,
                                                            dtype=tf.float32,                                                             
                                                            seed=None),
                                                            trainable = EMBEDDING_TRAIN,
                                                            name='embed_matrix')
            
            self.embed_en       = tf.nn.embedding_lookup(self.embed_matrix, self.encoder_inputs, name='embed_encoder')
            
            
    def _use_external_embedding(self):
        if self.use_glove == 1:
            print '[launch-text] use pre-trained embedding'
            self.embedding_init = self.embed_matrix.assign(self.embedding_placeholder)
            
            
    # cell instance
    def gru_cell(self):
        return tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)
    
    
    # cell instance with drop-out wrapper applied
    def gru_drop_out_cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell(), input_keep_prob=self.dr_prob, output_keep_prob=self.dr_prob)                    
    
    
    def test_cross_entropy_with_logit(self, logits, labels):
        x = logits
        z = labels
        return tf.maximum(x, 0) - x * z + tf.log(1 + tf.exp(-tf.abs(x)))
    
    
    def _create_gru_model(self):
        print '[launch-text] create gru cell'

        with tf.name_scope('text_RNN') as scope:
        
            with tf.variable_scope("text_GRU", reuse=False, initializer=tf.orthogonal_initializer()):
                
                cells_en = tf.contrib.rnn.MultiRNNCell( [ self.gru_drop_out_cell() for _ in range(self.num_layers) ] )

                (self.outputs_en, last_states_en) = tf.nn.dynamic_rnn(
                                                    cell=cells_en,
                                                    inputs= self.embed_en,
                                                    dtype=tf.float32,
                                                    sequence_length=self.encoder_seq,
                                                    time_major=False)
                
                self.final_encoder = last_states_en[-1]
                
        self.final_encoder_dimension   = self.hidden_dim
        
        
    def _create_output_layers(self):
        print '[launch-text] create output projection layer'        
        
        with tf.name_scope('text_output_layer') as scope:

            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, N_CATEGORY],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                 trainable=True,
                                                 name="similarity_matrix")
            
            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                 trainable=True,
                                                 name="output_bias")
            
            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b
        
        with tf.name_scope('loss') as scope:
            
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels )
            self.loss = tf.reduce_mean( self.batch_loss  )
                    
                
    def _create_output_layers_for_multi(self):
        print '[launch-text] create output projection layer for multi'        
        
        with tf.name_scope('text_output_layer') as scope:

            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, (self.final_encoder_dimension/2)],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                 trainable=True,
                                                 name="similarity_matrix")
            
            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                 trainable=True,
                                                 name="output_bias")
            
            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b
        
    
    def _create_optimizer(self):
        print '[launch-text] create optimizer'
        
        with tf.name_scope('text_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)
    
    
    def _create_summary(self):
        print '[launch-text] create summary'
        
        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    
    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._use_external_embedding()
        self._create_gru_model()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
