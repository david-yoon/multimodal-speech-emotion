###########################################
# SE train - Audio
# encoder_size = 750
###########################################
CUDA_VISIBLE_DEVICES=1 python train_audio.py --batch_size 128 --encoder_size 750 --num_layer 1 --hidden_dim 200 --lr=0.001 --num_train_steps 10000 --is_save 0 --dr 0.7 --graph_prefix 'SE_prosody' --data_path '../data/target/audio_woZ_set01/'


###########################################
# SE train - Text
# encoder_size = 128
###########################################
CUDA_VISIBLE_DEVICES=1 python train_text.py --batch_size 128 --encoder_size 128 --num_layer 1 --hidden_dim 200 --lr=0.001 --num_train_steps 10000 --is_save 0 --dr 0.3 --use_glove 1 --graph_prefix 'SE_nlp' --data_path '../data/target/audio_woZ_set01/'


###########################################
# SE train - Multi
# encoder_size_audio = 750
# encoder_size_text    = 128
###########################################
CUDA_VISIBLE_DEVICES=1 python train_multi.py --batch_size 128 --lr=0.001 --encoder_size_audio 750 --num_layer_audio 1 --hidden_dim_audio 200 --dr_audio 0.7 --encoder_size_text 128 --num_layer_text 1 --hidden_dim_text 200 --dr_text 0.3 --num_train_steps 10000 --is_save 0  --use_glove 1 --graph_prefix 'SE_multi' --data_path '../data/target/audio_woZ_set01/'


###########################################
# SE train - Multi Attn
# encoder_size_audio = 750
# encoder_size_text    = 128
# control = hop
###########################################
CUDA_VISIBLE_DEVICES=1 python train_multi_attn.py --batch_size 128 --lr=0.001 --encoder_size_audio 750 --num_layer_audio 1 --hidden_dim_audio 200 --dr_audio 0.7 --encoder_size_text 128 --num_layer_text 1 --hidden_dim_text 200 --dr_text 0.3 --num_train_steps 10000 --is_save 0  --use_glove 1 --graph_prefix 'SE_multi_attn_prosody' --data_path '../data/target/audio_woZ_set01/'

