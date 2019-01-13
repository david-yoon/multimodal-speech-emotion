#-*- coding: utf-8 -*-

"""
what    : process data, generate batch
"""

import numpy as np
import random

from project_config import *

class ProcessDataAudio:

    # store data
    train_set = []
    dev_set = []
    test_set = []
    
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        # load data
        self.train_set = self.load_data(DATA_TRAIN_MFCC, DATA_TRAIN_MFCC_SEQN, DATA_TRAIN_PROSODY, DATA_TRAIN_LABEL)
        self.dev_set  = self.load_data(DATA_DEV_MFCC, DATA_DEV_MFCC_SEQN, DATA_DEV_PROSODY, DATA_DEV_LABEL)
        self.test_set  = self.load_data(DATA_TEST_MFCC, DATA_TEST_MFCC_SEQN, DATA_TEST_PROSODY, DATA_TEST_LABEL)
       
        
    def load_data(self, audio_mfcc, mfcc_seqN, audio_prosody, label):
     
        print 'load data : ' + audio_mfcc + ' ' +  mfcc_seqN + ' ' + audio_prosody + ' ' + label
        output_set = []

        tmp_audio_mfcc          = np.load(self.data_path + audio_mfcc)
        tmp_mfcc_seqN          = np.load(self.data_path + mfcc_seqN)
        tmp_audio_prosody     = np.load(self.data_path + audio_prosody)
        tmp_label                   = np.load(self.data_path + label)

        for i in xrange( len(tmp_label) ) :
            output_set.append( [tmp_audio_mfcc[i], tmp_mfcc_seqN[i], tmp_audio_prosody[i], tmp_label[i]] )
        print '[completed] load data'
        
        return output_set
        
    
    """
        inputs: 
            data             : data to be processed (train/dev/test)
            batch_size    : mini-batch size
            encoder_size : max encoder time step
            
            is_test           : True, inference stage (ordered input)  ( default : False )
            start_index     : start index of mini-batch

        return:
            encoder_input   : [batch, time_step(==encoder_size), mfcc_dim]
            encoder_seq     : [batch] - valid mfcc step
            encoder_prosody         : [batch, prosody_dim] 
            labels               : [batch, category] - category is one-hot vector
    """
    def get_batch(self, data, batch_size, encoder_size, is_test=False, start_index=0):

        encoder_inputs, encoder_seq, encoder_prosody, labels = [], [], [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in xrange(batch_size):

            if is_test is False:
                # train case -  random sampling
                mfcc, seqN, prosody, label = random.choice(data)
                
            else:
                # dev, test case = ordered data
                if index >= len(data):
                    mfcc, seqN, prosody, label = data[0]  # won't be evaluated
                    index += 1
                else: 
                    mfcc, seqN, prosody, label = data[index]
                    index += 1

            # mix mfcc and prosody feature (not effective)
            # mfcc [seq_max x mfcc_dim],  prosody  [prosody_dim,]
            # audio_mix = [seq_max, mfcc+prosody]
            # prosody_tile = np.reshape(  np.tile(prosody,(N_SEQ_MAX)), (N_SEQ_MAX, N_AUDIO_PROSODY) )
            # audio_mix = np.concatenate( (mfcc,prosody_tile), axis=1 )
                
            audio_mix = mfcc
            
            encoder_inputs.append( audio_mix[:encoder_size] )
            encoder_seq.append( np.min((seqN,encoder_size)) )
            
            encoder_prosody.append( prosody )
            
            tmp_label = np.zeros( N_CATEGORY, dtype=np.float )
            tmp_label[label] = 1
            labels.append( tmp_label )
            
        return encoder_inputs, encoder_seq, encoder_prosody, np.reshape( labels, (batch_size,N_CATEGORY) )
    
    