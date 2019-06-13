#-*- coding: utf-8 -*-

"""
what    : evaluation
"""

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from scipy.stats import rankdata

from project_config import *

"""
    desc  : 
    
    inputs: 
        sess  : tf session
        model : model for test
        data  : such as the dev_set, test_set...
            
    return:
        sum_batch_ce : sum cross_entropy
        accr         : accuracy
        
"""
def run_test(sess, model, batch_gen, data):
    
    list_batch_ce = []
    list_batch_correct = []
    
    list_pred = []
    list_label = []
    
    bpred, bloss = [], []

    max_loop  = len(data) / model.batch_size
    remaining = len(data) % model.batch_size

    # evaluate data ( N of chunk (batch_size) + remaining( +1) )
    for test_itr in xrange( max_loop + 1 ):
        
        raw_encoder_inputs_audio, raw_encoder_seq_audio, raw_encoder_prosody, raw_encoder_inputs_text, raw_encoder_seq_text, raw_label = batch_gen.get_batch(
                            batch_size=model.batch_size,
                            data=data,
                            encoder_size_audio=model.encoder_size_audio,
                            encoder_size_text=model.encoder_size_text,
                            is_test=True,
                            start_index= (test_itr* model.batch_size)
                            )
        
        # prepare data which will be push from pc to placeholder
        input_feed = {}

        input_feed[model.encoder_inputs_audio] = raw_encoder_inputs_audio
        input_feed[model.encoder_seq_audio] = raw_encoder_seq_audio
        input_feed[model.encoder_prosody] = raw_encoder_prosody
        input_feed[model.dr_prob_audio] = 1.0              # no drop out while evaluating

        input_feed[model.encoder_inputs_text] = raw_encoder_inputs_text
        input_feed[model.encoder_seq_text] = raw_encoder_seq_text
        input_feed[model.dr_prob_text] = 1.0                # no drop out while evaluating

        input_feed[model.y_labels] = raw_label
    
        try:
            bpred, bloss = sess.run([model.batch_pred, model.batch_loss], input_feed)
        except:
            print "excepetion occurs in valid step : " + str(test_itr)
            pass
        
        # remaining data case (last iteration)
        if test_itr == (max_loop):
            bpred = bpred[:remaining]
            bloss = bloss[:remaining]
            raw_label = raw_label[:remaining]
        
        # batch loss
        list_batch_ce.extend( bloss )
        
        # batch accuracy
        list_pred.extend( np.argmax(bpred, axis=1) )
        list_label.extend( np.argmax(raw_label, axis=1) )
      
    
    if IS_LOGGING:
        with open( '../analysis/inference_log/mult.txt', 'w' ) as f:
            f.write( ' '.join( [str(x) for x in list_pred] ) )
        
        with open( '../analysis/inference_log/mult_label.txt', 'w' ) as f:
            f.write( ' '.join( [str(x) for x in list_label] ) )
        
        
    list_batch_correct = [1 for x, y in zip(list_pred,list_label) if x==y]
    
    sum_batch_ce = np.sum( list_batch_ce )
    accr = np.sum ( list_batch_correct ) / float( len(data) )
    
    value1 = summary_pb2.Summary.Value(tag="valid_loss", simple_value=sum_batch_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy", simple_value=accr )
    summary = summary_pb2.Summary(value=[value1, value2])
    
    return sum_batch_ce, accr, summary, list_pred
    
