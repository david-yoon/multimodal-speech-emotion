#-*- coding: utf-8 -*-

import tensorflow as tf

    
'''
desc : apply luong attention to target vector with given condition

input :
   - batch_size             : 
   - target                 : [batch, seq, embed]
   - condition              : [batch, embed] --> last hidden
   - batch_seq              : [batch] valid encoder seq
   - max_len                : max encoder length
   - hidden                 : should be same btw target and condition, otherwise code should be changed

output : 
   - attented target : weighted sum [batch, embed]
   - norm_dot : attention weight
'''
def luong_attention( batch_size, target, condition, batch_seq, max_len, hidden_dim ) :

    # same dim [batch, max_seq, embed]
    batch_seq_embed_target = tf.reshape( target, [batch_size, max_len, hidden_dim] )
    

    batch_embed_given = condition
    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )

    # calculate similarity 
    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )
    dot = tf.squeeze(dot)
    
    
    """
    # pad 부분을 -inf 값으로 대체 --> 그래야 softmax 후 0으로 떨어짐
    """
    mask = tf.sequence_mask( lengths=batch_seq, maxlen=max_len, dtype=tf.float32 )
    mask_value = -tf.ones_like( mask ) * tf.float32.max
    mask_value = tf.multiply( mask_value, ( 1- mask ) )
    base = mask_value
    
    norm_dot = tf.nn.softmax( dot + base, axis=-1 )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( batch_seq_embed_target, tf.expand_dims(norm_dot, -1) )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )
    
    return weighted_sum, norm_dot
    
    
'''
desc : apply luong attention to target vector with given condition

To be modified

input :
   - batch_size             : 
   - target                 : [batch, seq, embed]
   - condition              : [batch, embed] --> last hidden
   - target_encoder_length  : max encoder length
   - hidden                 : should be same btw target and condition, otherwise code should be changed

output : 
   - attented target : weighted sum [batch, embed]
   - norm_dot : attention weight
'''
def luong_attention_mul_condition( batch_size, target, condition, target_dim, condition_dim, max_target_encoder_length) :

    weighted_sum = 0
    norm_dot = 0
  
    W_matrix = tf.Variable(tf.random_uniform([target_dim, condition_dim],
                                                  minval= -0.25,
                                                  maxval= 0.25,
                                                  dtype=tf.float32,
                                                  seed=None),
                                               trainable=True,
                                               name="attn_W_target")
    
    attn_bias = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                 trainable=True,
                                                 name="attn_bias")
    
    
    W_matrix = tf.reshape( W_matrix, [1, target_dim, condition_dim])
    W_matrix = tf.tile( W_matrix, [batch_size, 1, 1])
    
    tmp_target = tf.matmul( target, W_matrix )
    
    dot = tf.multiply(tmp_target, condition)
    dot = tf.reduce_sum(dot, axis=2) + attn_bias
    
    # pad 부분을 작은 값으로 대체 --> 그래야 softmax 후 0으로 떨어짐
    pad_position = tf.equal(tf.reshape(dot, [batch_size, max_target_encoder_length]), 0.0)
    base = tf.to_float(pad_position) * -1e9
    
    norm_dot = tf.nn.softmax( dot+base, dim=1 )
    norm_dot = tf.reshape( norm_dot, [batch_size, max_target_encoder_length, 1] )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )

    return weighted_sum, norm_dot, W_matrix, tmp_target, dot
