#-*- coding: utf-8 -*-



import tensorflow as tf



'''

desc : apply luong attention to target vector with given condition

input :

   - batch_size             : 

   - target                 : [batch, seq, embed]

   - condition              : [batch, embed] --> last hidden

   - target_encoder_length  : max encoder length

   - hidden                 : should be same btw target and condition, otherwise code should be changed

output : 

   - attented target : weighted sum [batch, embed]

'''

def luong_attention( batch_size, target, condition, target_encoder_length, hidden_dim ) :



    # same dim [batch, max_seq, embed]

    batch_seq_embed_target = tf.reshape( target, [batch_size, target_encoder_length, hidden_dim] )

    

    batch_embed_given = condition

    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )

    

    # calculate similarity 

    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )

    norm_dot = tf.nn.softmax( dot, dim=1 )

    

    # weighted sum by using similarity (normalized)

    target_mul_norm = tf.multiply( batch_seq_embed_target, norm_dot )

    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )



    return weighted_sum
