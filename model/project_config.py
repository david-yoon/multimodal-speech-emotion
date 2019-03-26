
################################
#     Training             
################################
CAL_ACCURACY_FROM = 0
MAX_EARLY_STOP_COUNT = 7
EPOCH_PER_VALID_FREQ = 0.3

DATA_TRAIN_MFCC            = 'train_audio_mfcc.npy'
DATA_TRAIN_MFCC_SEQN  = 'train_seqN.npy'
DATA_TRAIN_PROSODY      = 'train_audio_prosody.npy'
#DATA_TRAIN_PROSODY      = 'train_audio_emobase2010.npy'   # easy emobase2010 setting
DATA_TRAIN_LABEL           = 'train_label.npy'
DATA_TRAIN_TRANS          = 'train_nlp_trans.npy'


DATA_DEV_MFCC              = 'dev_audio_mfcc.npy'
DATA_DEV_MFCC_SEQN    = 'dev_seqN.npy'
DATA_DEV_PROSODY        = 'dev_audio_prosody.npy'
#DATA_DEV_PROSODY        = 'dev_audio_emobase2010.npy'   # easy emobase2010 setting
DATA_DEV_LABEL             = 'dev_label.npy'
DATA_DEV_TRANS            = 'dev_nlp_trans.npy'


DATA_TEST_MFCC            = 'test_audio_mfcc.npy'
DATA_TEST_MFCC_SEQN  = 'test_seqN.npy'
DATA_TEST_PROSODY      = 'test_audio_prosody.npy'
#DATA_TEST_PROSODY     = 'test_audio_emobase2010.npy'   # easy emobase2010 setting
DATA_TEST_LABEL           = 'test_label.npy'
DATA_TEST_TRANS          = 'test_nlp_trans.npy'


DIC                               = 'dic.pkl'
GLOVE                              = 'W_embedding.npy'


################################
#     Audio
################################
N_CATEGORY = 4
N_AUDIO_MFCC = 39
N_AUDIO_PROSODY= 35
#N_AUDIO_PROSODY= 1582   # easy emobase2010 setting
N_SEQ_MAX = 750                 # max 1,000 (MSP case only)


################################
#     NLP
################################
N_SEQ_MAX_NLP = 128
DIM_WORD_EMBEDDING = 100   # when using glove it goes to 300 automatically
EMBEDDING_TRAIN = True           # True is better



################################
#     ETC
################################
IS_LOGGING = False