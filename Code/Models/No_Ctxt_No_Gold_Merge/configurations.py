# model params
VOCAB_SIZE = 50000    # 1 for UNK
MAX_RESP_LEN = 50
MAX_CTX_LEN = 500

# training details
TRAIN_SIZE = -1    # -1 => train on all
BATCH_SIZE = 128

# hpc file paths
train_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/train.txt'
val_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/val.txt'
test_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetNewPruned11M/test.txt'
count_vect_vocab_file = '/home/cse/dual/cs5130275/DERP/Code/Models/LogisticRegBaseline/vocab_50k'
save_model_path = '/scratch/cse/dual/cs5130275/DERP/Models/No_Ctxt_No_Gold_Merge/' + 'final_layer_merge'
load_model_path = ''
# load_model_path = '/scratch/cse/dual/cs5130275/DERP/Models/GRU_Ctxt_GRU_Resp/' + 'CTXT_HIDDEN_STATE_' + str(CTXT_GRU_HIDDEN_STATE) + '_RESP_HIDDEN_STATE_' + str(RESP_GRU_HIDDEN_STATE) + '_VOCAB_SIZE_' + str(VOCAB_SIZE) + '_MAX_RESP_LEN_' + str(MAX_RESP_LEN) + '_EMBEDDING_DIM_' + str(EMBEDDING_DIM) + '_DENSE_HIDDEN_STATE_' + str(DENSE_HIDDEN_STATE) + '_BATCH_SIZE_' + str(BATCH_SIZE) + '_val500_on_epoch_16.weights'