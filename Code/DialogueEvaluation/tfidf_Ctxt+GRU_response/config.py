# model params
GRU_HIDDEN_STATE = 300
VOCAB_SIZE = 50000    # 1 for UNK
MAX_RESP_LEN = 50
EMBEDDING_DIM = 300
DENSE_HIDDEN_STATE1 = 300
DENSE_HIDDEN_STATE2 = 100

# training details
TRAIN_SIZE = -1    # -1 => train on all
BATCH_SIZE = 512
DROPOUT = 0.1

# hpc file paths
train_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/train.txt'
val_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/val.txt'
test_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/test.txt'
train_ctxt_tfidfed_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/train_ctxt_count_vect_tfidfed.npz'
train_ctxt_preprocessed_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/train_ctxt_idxd.json.bz2'
train_gold_resp_preprocessed_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/train_gold_resp_idxd.json.bz2'
train_alt_resp_preprocessed_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/train_alt_resp_idxd.json.bz2'
count_vect_vocab_file = '/home/cse/dual/cs5130275/DERP/Code/Models/LogisticRegBaseline/vocab_50k'
tfidf_transformer_file = '/home/cse/dual/cs5130275/DERP/Code/Models/LogisticRegBaseline/tfidf_transformer_50k'
model_name = 'GRU_HIDDEN_STATE_' + str(GRU_HIDDEN_STATE) + '_VOCAB_SIZE_' + str(VOCAB_SIZE) + '_MAX_RESP_LEN_' + str(MAX_RESP_LEN) + '_EMBEDDING_DIM_' + str(EMBEDDING_DIM) + '_DENSE_HIDDEN_STATE1_' + str(DENSE_HIDDEN_STATE1) + '_DENSE_HIDDEN_STATE2_' + str(DENSE_HIDDEN_STATE2) + '_DROPOUT_' + str(DROPOUT) + '_BATCH_SIZE_' + str(BATCH_SIZE)                  
save_model_path = '/scratch/cse/dual/cs5130275/DERP/Models/tfidf_Ctxt+GRU_response/' + model_name + '.weights.hdf5'                                                     
load_model_path = ''
save_pred_path = './'

# local file paths
# train_file = 'train100.txt'
# val_file = 'train100.txt'
# count_vect_vocab_file = '../LogisticRegBaseline/vocab_50k'
# tfidf_transformer_file = '../LogisticRegBaseline/tfidf_transformer_50k'
