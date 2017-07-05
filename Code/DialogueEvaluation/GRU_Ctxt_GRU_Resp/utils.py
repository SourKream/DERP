from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')
from base_utils import *

def generate_X_Y(options,key='TRAIN', num_elem = -1):  
    X = []
    y = []
    with open(options[key],'rb') as f:
        for line in f:
            line = line.strip().split('\t')            
            X.append(line[:3])
            y.append(int(line[3]))            
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
            cur_batch_ctxt, cur_batch_gold_resp, cur_batch_alt_resp = [list(x) for x in zip(*cur_batch_x)]
            
            for idx in xrange(len(cur_batch_ctxt)):
                cur_batch_ctxt[idx] = [ map_to_ix(word, options['VOCAB'], lower=options['LOWER']) for word in my_tokenize(cur_batch_ctxt[idx])]
                cur_batch_gold_resp[idx] = [ map_to_ix(word, options['VOCAB'], lower=options['LOWER']) for word in my_tokenize(cur_batch_gold_resp[idx])]
                cur_batch_alt_resp[idx] = [ map_to_ix(word, options['VOCAB'], lower=options['LOWER']) for word in my_tokenize(cur_batch_alt_resp[idx])]
            
            cur_batch_ctxt_vec = pad_sequences(cur_batch_ctxt, maxlen=MAX_CTX_LEN, value=0, padding='post', truncating='post')
            cur_batch_gold_resp_vec = pad_sequences(cur_batch_gold_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            cur_batch_alt_resp_vec = pad_sequences(cur_batch_alt_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            
            yield [cur_batch_ctxt_vec, cur_batch_gold_resp_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)

def get_best_model(file_prefix, model):
    model_file =  get_best_model_file(file_prefix, "max")
    if model_file.endswith('.hdf5'):
        model.load_weights(model_file, by_name = True)
    else:
        weights = joblib.load(model_file)
        model.set_weights(weights)
    return model

def get_probs(options, X, y, generator, model):    
    gen_X_Y = generator(X,y, options)
    steps_X = len(X) / options['BATCH_SIZE'] if len(X) % options['BATCH_SIZE'] == 0 else ((len(X) / options['BATCH_SIZE']) ) + 1
    preds = model.predict_generator(gen_X_Y, steps = steps_X, verbose=1).flatten()
    return preds
    
