#-*- coding: utf-8 -*-

"""
what    : Single Encoder Model for Multi (Audio + Text) with attention
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from project_config import *

from SE_model_audio import *
from SE_model_text import *
from model_util import luong_attention


class SingleEncoderModelMultiAttn:
    
    def __init__(self,
                 batch_size,
                 lr,
                 encoder_size_audio,  # for audio
                 num_layer_audio,
                 hidden_dim_audio,
                 dr_audio,
                 dic_size,             # for text
                 use_glove,
                 encoder_size_text,
                 num_layer_text,
                 hidden_dim_text,
                 dr_text
                ):

        # for audio
        self.encoder_size_audio = encoder_size_audio
        self.num_layers_audio = num_layer_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.dr_audio = dr_audio
        
        self.encoder_inputs_audio = []
        self.encoder_seq_length_audio =[]
        
        # for text        
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.encoder_size_text = encoder_size_text
        self.num_layers_text = num_layer_text
        self.hidden_dim_text = hidden_dim_text
        self.dr_text = dr_text
        
        self.encoder_inputs_text = []
        self.encoder_seq_length_text =[]

        # common        
        self.batch_size = batch_size
        self.lr = lr
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
        print '[launch-multi] placeholders'
        with tf.name_scope('multi_placeholder'):
            
            # for audio
            self.encoder_inputs_audio  = self.model_audio.encoder_inputs  # [batch, time_step, audio]
            self.encoder_seq_audio     = self.model_audio.encoder_seq
            self.encoder_prosody       = self.model_audio.encoder_prosody
            self.dr_prob_audio         = self.model_audio.dr_prob
            
            # for text
            self.encoder_inputs_text  = self.model_text.encoder_inputs
            self.encoder_seq_text     = self.model_text.encoder_seq
            self.dr_prob_text         = self.model_text.dr_prob

            # common
            self.y_labels             = tf.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")
            
            # for using pre-trained embedding
            self.embedding_placeholder = self.model_text.embedding_placeholder


    def _create_model_audio(self):
        print '[launch-multi] create audio model'
        self.model_audio =  SingleEncoderModelAudio(
                                                        batch_size=self.batch_size,
                                                        encoder_size=self.encoder_size_audio,
                                                        num_layer=self.num_layers_audio,
                                                        hidden_dim=self.hidden_dim_audio,
                                                        lr = self.lr,
                                                        dr= self.dr_audio
                                                        )
        self.model_audio._create_placeholders()
        self.model_audio._create_gru_model()
        self.model_audio._add_prosody()
        #self.model_audio._create_output_layers_for_multi()
        


    def _create_model_text(self):
        print '[launch-multi] create text model'        
        self.model_text = SingleEncoderModelText(
                                                    batch_size=self.batch_size,
                                                    dic_size=self.dic_size,
                                                    use_glove=self.use_glove,
                                                    encoder_size=self.encoder_size_text,
                                                    num_layer=self.num_layers_text,
                                                    hidden_dim=self.hidden_dim_text,
                                                    lr = self.lr,
                                                    dr= self.dr_text
                                                )
        
        self.model_text._create_placeholders()
        self.model_text._create_embedding()
        self.model_text._use_external_embedding()
        self.model_text._create_gru_model()
        #self.model_text._create_output_layers_for_multi()


    def _create_attention_module(self):
        print '[launch-multi] create attention module'
        # project audio dimension_size to text dimension_size
        self.attnM = tf.Variable(tf.random_uniform([self.model_audio.final_encoder_dimension, self.model_text.final_encoder_dimension],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                 trainable=True,
                                                 name="attn_projection_helper")
            
        self.attnb = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                 trainable=True,
                                                 name="attn_bias")
        

        self.attn_audio_final_encoder = tf.matmul(self.model_audio.final_encoder, self.attnM) + self.attnb
        
        self.final_encoder = luong_attention (
                                                batch_size = self.batch_size,
                                                target = self.model_text.outputs_en,
                                                condition = self.attn_audio_final_encoder,
                                                target_encoder_length = self.model_text.encoder_size,
                                                hidden_dim = self.model_text.final_encoder_dimension
                                            )

        
    def _create_output_layers(self):
        print '[launch-multi] create output projection layer from (text_final_dim(==audio) + text_final_dim)'
        
        with tf.name_scope('multi_output_layer') as scope:

            self.final_encoder = tf.concat( [self.final_encoder, self.attn_audio_final_encoder], axis=1 )
            
            self.M = tf.Variable(tf.random_uniform([(self.model_text.final_encoder_dimension)+(self.model_text.final_encoder_dimension), N_CATEGORY],
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

    
    def _create_optimizer(self):
        print '[launch-multi] create optimizer'
        
        with tf.name_scope('multi_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)
    
    
    def _create_summary(self):
        print '[launch-multi] create summary'
        
        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    
    def build_graph(self):
        self._create_model_audio()
        self._create_model_text()
        self._create_placeholders()
        self._create_attention_module()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
