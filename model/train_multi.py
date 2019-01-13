#-*- coding: utf-8 -*-

"""
what    : train multi model (audio + text)
"""
import tensorflow as tf
import os
import time
import argparse
import datetime

from SE_model_multi import *
from process_data_multi import *
from evaluation_multi import *
from project_config import *


# for training         
def train_step(sess, model, batch_gen):
    raw_encoder_inputs_audio, raw_encoder_seq_audio, raw_encoder_prosody, raw_encoder_inputs_text, raw_encoder_seq_text, raw_label = batch_gen.get_batch(
                            batch_size=model.batch_size,
                            data=batch_gen.train_set,
                            encoder_size_audio=model.encoder_size_audio,
                            encoder_size_text=model.encoder_size_text,
                            is_test=False
                            )

    # prepare data which will be push from pc to placeholder
    input_feed = {}
    
    input_feed[model.encoder_inputs_audio] = raw_encoder_inputs_audio
    input_feed[model.encoder_seq_audio] = raw_encoder_seq_audio
    input_feed[model.encoder_prosody] = raw_encoder_prosody
    input_feed[model.dr_prob_audio] = model.dr_audio
    
    input_feed[model.encoder_inputs_text] = raw_encoder_inputs_text
    input_feed[model.encoder_seq_text] = raw_encoder_seq_text
    input_feed[model.dr_prob_text] = model.dr_text
    
    input_feed[model.y_labels] = raw_label
    
    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
    
    return summary

    
def train_model(model, batch_gen, num_train_steps, valid_freq, is_save=0, graph_dir_name='default'):
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    summary = None
    val_summary = None
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        early_stop_count = MAX_EARLY_STOP_COUNT
        
        # for text model - GLOVE
        if model.use_glove == 1:
            sess.run(model.model_text.embedding_init, feed_dict={ model.embedding_placeholder: batch_gen.get_glove() })
            print '[completed] loading pre-trained embedding vector to placeholder'
            
        
        # if exists check point, starts from the check point
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('save/' + graph_dir_name + '/'))
        if ckpt and ckpt.model_checkpoint_path:
            print ('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        writer = tf.summary.FileWriter('./graph/'+graph_dir_name, sess.graph)

        initial_time = time.time()
        
        min_ce = 1000000
        best_dev_accr = 0
        test_accr_at_best_dev = 0
        
        for index in xrange(num_train_steps):

            try:
                # run train 
                summary = train_step(sess, model, batch_gen)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except:
                print "excepetion occurs in train step"
                pass
                
            
            # run validation
            if (index + 1) % valid_freq == 0:
                
                dev_ce, dev_accr, dev_summary, _ = run_test(sess=sess,
                                                         model=model, 
                                                         batch_gen=batch_gen,
                                                         data=batch_gen.dev_set)
                
                writer.add_summary( dev_summary, global_step=model.global_step.eval() )
                
                end_time = time.time()

                if index > CAL_ACCURACY_FROM:

                    if ( dev_ce < min_ce ):
                        min_ce = dev_ce

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )

                        early_stop_count = MAX_EARLY_STOP_COUNT
                        
                        test_ce, test_accr, _, _ = run_test(sess=sess,
                                                         model=model,
                                                         batch_gen=batch_gen,
                                                         data=batch_gen.test_set)
                        
                        best_dev_accr = dev_accr
                        test_accr_at_best_dev = test_accr
                                                                                            
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print "early stopped"
                            break
                             
                        test_accr = 0
                        early_stop_count = early_stop_count -1
                        
                    print str( int(end_time - initial_time)/60 ) + " mins" + \
                        " step/seen/itr: " + str( model.global_step.eval() ) + "/ " + \
                                               str( model.global_step.eval() * model.batch_size ) + "/" + \
                                               str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + \
                        "\tdev: " + '{:.3f}'.format(dev_accr)  + "  test: " + '{:.3f}'.format(test_accr) + "  loss: " + '{:.2f}'.format(dev_ce) 
                
        writer.close()
            
        print ('Total steps : {}'.format(model.global_step.eval()) )
        
        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    batch_gen.data_path.split('/')[-2] + '\t' + \
                    graph_dir_name + '\t' + str(best_dev_accr) + '\t' + str(test_accr_at_best_dev) + '\n')


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(data_path, batch_size, lr,
         encoder_size_audio, num_layer_audio, hidden_dim_audio, dr_audio,
         encoder_size_text, num_layer_text, hidden_dim_text, dr_text,
         use_glove,
         num_train_steps, is_save, graph_dir_name
         ):
    
    if is_save is 1:
        create_dir('save/')
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/'+graph_dir_name)
    
    batch_gen = ProcessDataMulti(data_path)
    
    model = SingleEncoderModelMulti(
                            batch_size = batch_size,
                            lr = lr,
                            encoder_size_audio = encoder_size_audio,
                            num_layer_audio = num_layer_audio,
                            hidden_dim_audio = hidden_dim_audio,                
                            dr_audio = dr_audio,
                            encoder_size_text = encoder_size_text,
                            num_layer_text = num_layer_text,
                            hidden_dim_text = hidden_dim_text,                
                            dr_text = dr_text,
                            dic_size = batch_gen.dic_size,
                            use_glove = use_glove
                            )

    model.build_graph()
    
    valid_freq = int( len(batch_gen.train_set) * EPOCH_PER_VALID_FREQ / float(batch_size)  ) + 1
    print "[Info] Valid Freq = " + str(valid_freq)

    train_model(model, batch_gen, num_train_steps, valid_freq, is_save, graph_dir_name)
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-1)
    
    p.add_argument('--encoder_size_audio', type=int, default=750)
    p.add_argument('--num_layer_audio', type=int, default=1)
    p.add_argument('--hidden_dim_audio', type=int, default=50)
    p.add_argument('--dr_audio', type=float, default=1.0)
    
    p.add_argument('--use_glove', type=int, default=0)
    p.add_argument('--encoder_size_text', type=int, default=750)
    p.add_argument('--num_layer_text', type=int, default=1)
    p.add_argument('--hidden_dim_text', type=int, default=50)
    p.add_argument('--dr_text', type=float, default=1.0)
    
    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    

    embed_train = ''
    if EMBEDDING_TRAIN == False:
        embed_train = 'F'
    
    args = p.parse_args()
    
    graph_name = args.graph_prefix + \
                    '_D' + (args.data_path).split('/')[-2] + \
                    '_b' + str(args.batch_size) + \
                    '_esA' + str(args.encoder_size_audio) + \
                    '_LA' + str(args.num_layer_audio) + \
                    '_HA' + str(args.hidden_dim_audio) + \
                    '_drA' + str(args.dr_audio) + \
                    '_esT' + str(args.encoder_size_text) + \
                    '_LT' + str(args.num_layer_text) + \
                    '_HT' + str(args.hidden_dim_text) + \
                    '_G' + str(args.use_glove) + embed_train + \
                    '_drT' + str(args.dr_text)
    
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")
    
    main(
        data_path = args.data_path,
        batch_size = args.batch_size,
        lr = args.lr,
        encoder_size_audio = args.encoder_size_audio,
        num_layer_audio = args.num_layer_audio,
        hidden_dim_audio = args.hidden_dim_audio,
        dr_audio = args.dr_audio,
        
        use_glove = args.use_glove,
        encoder_size_text = args.encoder_size_text,
        num_layer_text = args.num_layer_text,
        hidden_dim_text = args.hidden_dim_text,
        dr_text = args.dr_text,
        
        num_train_steps = args.num_train_steps,
        is_save = args.is_save,
        graph_dir_name = graph_name
        )