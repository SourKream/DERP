from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')
from base_utils import *
import h5py
def generate_X_Y(options,key='TRAIN', num_elem = -1):  
    X = []
    y = []
    with open(options[key],'rb') as f:
        for line in f:
            line = line.strip().split('\t')
            X.append(line[:3])
            y.append(options['MARGIN'])
            if len(X) == num_elem:
                assert len(y) == len(X)
                break
    return X,y

def data_generator(data_x, data_y, options):
    BATCH_SIZE = options['BATCH_SIZE']
    MAX_RESP_LEN = options['MAX_RESP_LEN']
    MAX_CTX_LEN = options['MAX_CTX_LEN']
    while True:
        for i in xrange(0, len(data_x), BATCH_SIZE):
            cur_batch_x = data_x[i:i+BATCH_SIZE]
            cur_batch_y = data_y[i:i+BATCH_SIZE]            
            cur_batch_ctxt, cur_batch_pos_resp, cur_batch_neg_resp = [list(x) for x in zip(*cur_batch_x)]
            for idx in xrange(len(cur_batch_ctxt)):
                cur_batch_ctxt[idx] = [ map_to_ix(word, options['VOCAB'], lower=options['LOWER']) for word in my_tokenize(cur_batch_ctxt[idx])]
                cur_batch_pos_resp[idx] = [ map_to_ix(word, options['VOCAB'], lower=options['LOWER']) for word in my_tokenize(cur_batch_pos_resp[idx]) ]
                cur_batch_neg_resp[idx] = [ map_to_ix(word, options['VOCAB'], lower=options['LOWER']) for word in my_tokenize(cur_batch_neg_resp[idx]) ]
            cur_batch_ctxt_vec = pad_sequences(cur_batch_ctxt, maxlen=MAX_CTX_LEN, value=0, padding='post', truncating='post')
            cur_batch_pos_resp_vec = pad_sequences(cur_batch_pos_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            cur_batch_neg_resp_vec = pad_sequences(cur_batch_neg_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
    
            yield [cur_batch_ctxt_vec, cur_batch_pos_resp_vec, cur_batch_neg_resp_vec], np.array(cur_batch_y) 

def get_best_model(file_prefix, model):
    model_file =  get_best_model_file(file_prefix, "min")
    if model_file.endswith('.hdf5'):
        # model.load_weights(model_file, by_name = True)
        weights = convert_weights_from_hdf5(model_file)
        model.set_weights(weights)
    else:
        weights = joblib.load(model_file)
        model.set_weights(weights)
    return model
    
def convert_weights_from_hdf5(weights_hdf5_file):
    weights_hdf5 = h5py.File(weights_hdf5_file, mode='r')

    new_weights = []
    new_weights.append(weights_hdf5['model_1']['embedding_1']['embeddings'][:])

    new_weights.append(weights_hdf5['model_1']['bidirectional_1']['forward_gru_1']['kernel'][:]) #(300, 1536) 
    new_weights.append(weights_hdf5['model_1']['bidirectional_1']['forward_gru_1']['recurrent_kernel'][:]) #(512, 1536) 
    new_weights.append(weights_hdf5['model_1']['bidirectional_1']['forward_gru_1']['bias'][:]) #(1536,) 

    new_weights.append(weights_hdf5['model_1']['bidirectional_1']['backward_gru_1']['kernel'][:]) #(300, 1536) 
    new_weights.append(weights_hdf5['model_1']['bidirectional_1']['backward_gru_1']['recurrent_kernel'][:]) #(512, 1536)
    new_weights.append(weights_hdf5['model_1']['bidirectional_1']['backward_gru_1']['bias'][:]) #(1536,)

    new_weights.append(weights_hdf5['model_1']['bidirectional_2']['forward_gru_2']['kernel'][:]) #(300, 900)
    new_weights.append(weights_hdf5['model_1']['bidirectional_2']['forward_gru_2']['recurrent_kernel'][:]) #(300, 900)
    new_weights.append(weights_hdf5['model_1']['bidirectional_2']['forward_gru_2']['bias'][:]) #(900,)

    new_weights.append(weights_hdf5['model_1']['bidirectional_2']['backward_gru_2']['kernel'][:]) #(300, 900)
    new_weights.append(weights_hdf5['model_1']['bidirectional_2']['backward_gru_2']['recurrent_kernel'][:]) #(300, 900)
    new_weights.append(weights_hdf5['model_1']['bidirectional_2']['backward_gru_2']['bias'][:]) #(900,)

    new_weights.append(weights_hdf5['model_1']['bilinear_layer_1']['ctx_dense'][:]) # (1024, 600)
    
    new_weights.append(weights_hdf5['model_1']['dense_1']['kernel'][:]) #BI-MATCHING : (1625,100)
    new_weights.append(weights_hdf5['model_1']['dense_1']['bias'][:]) #(100,)

    new_weights.append(weights_hdf5['model_1']['dense_2']['kernel'][:]) #(100, 1)
    new_weights.append(weights_hdf5['model_1']['dense_2']['bias'][:]) #(1,)

    return new_weights



