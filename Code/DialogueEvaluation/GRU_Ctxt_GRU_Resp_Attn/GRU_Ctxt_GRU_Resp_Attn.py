from sklearn.externals import joblib
import keras
from keras.layers import Input, GRU, Embedding, Dense, Dropout 
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from keras import optimizers
from configurations import *
from utils import *
import theano.tensor as T
import argparse

from keras.engine.topology import Layer
from sklearn.externals import joblib
import numpy as np
import pdb

import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')

from base_utils import ModelCheckpoint

def get_arguments():
    parser = argparse.ArgumentParser(description='With attention model')
    parser.add_argument('-bad_grammar', action="store", default="False", dest="bad_grammar", type=str)
    parser.add_argument('-resume_training', action="store", default="False", dest="resume_training", type=str)
    opts = parser.parse_args(sys.argv[1:])  
    opts.bad_grammar = True if opts.bad_grammar == "True" else False
    opts.resume_training = True if opts.resume_training == "True" else False
    return opts

class AttentionLayer(Layer):
    def __init__(self, dense_dim, bilinear = False, return_att=False, **kwargs):
        self.return_att = return_att
        self.bilinear = bilinear
        self.dense_dim = dense_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):       
        self.ctxt_dim = input_shape[0][2]
        self.resp_dim = input_shape[1][1]
        if self.bilinear :
            self.ctxt_dense = self.add_weight(name='ctxt_dense', shape=(self.ctxt_dim, self.resp_dim), initializer='glorot_uniform', trainable=True)
        else:
            self.ctxt_dense = self.add_weight(name='ctxt_dense', shape=(self.ctxt_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
            self.resp_dense = self.add_weight(name='resp_dense', shape=(self.resp_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
            self.alpha_dense = self.add_weight(name='alpha_dense', shape=(self.dense_dim, 1), initializer='glorot_uniform', trainable=True) # n_dim x 1
        super(AttentionLayer, self).build(input_shape)

    def call(self, input_tensors, mask = None):
        ''' wbw attention layer:
        :param ctxt (input_tensors[0]) : batch_size x T x ctxt_dim
        :param resp (input_tensors[1]) : batch_size x resp_dim
        '''
        ctxt = input_tensors[0]
        resp = input_tensors[1]
        ctxt_mask = None
        if not mask is None:
            ctxt_mask = mask[0]     
        if self.bilinear:
            ctxt_w = T.dot(ctxt, self.ctxt_dense)
            unnorm_alpha = T.batched_dot(ctxt_w, resp.dimshuffle((0,1,'x'))).flatten(2) # batch_size x T        
        else:
            resp_w = T.dot(resp, self.resp_dense)
            ctxt_w = T.dot(ctxt, self.ctxt_dense) # bt_sz x T x dense_dim
            resp_w_rep = resp_w[:,None,:] # bt_sz x T x dense_dim
            pre_alpha = T.tanh(ctxt_w + resp_w_rep) # bt_sz x T x dense_dim
            unnorm_alpha = T.dot(pre_alpha, self.alpha_dense).flatten(2) # bt_sz x T

        if not ctxt_mask is None:
            unnorm_alpha_masked = unnorm_alpha - 1000 * (1. - ctxt_mask)
        else:
            unnorm_alpha_masked = unnorm_alpha
        alpha = T.nnet.softmax(unnorm_alpha_masked) # bt_sz x T
        attended_ctxt = T.batched_dot(alpha.dimshuffle((0,'x',1)), ctxt)[:,0,:] # bt_sz x ctxt_dim
        if self.return_att:
            return [attended_ctxt, alpha]
        else:
            return attended_ctxt

    def compute_output_shape(self, input_shape):
        if self.return_att:
            return [(input_shape[0][0], input_shape[0][2]), (input_shape[0][0], input_shape[0][1])]
        else:
            return (input_shape[0][0], input_shape[0][2])

    def compute_mask(self, input_tensors, input_masks):
        if self.return_att:
            return [None, None]
        else:
            return None     

def create_model(options):
    ctxt = Input(shape=(options['MAX_CTX_LEN'],))
    gold_resp = Input(shape=(options['MAX_RESP_LEN'],))
    alt_resp = Input(shape=(options['MAX_RESP_LEN'],))  
    embedding = None
    if options['USE_GLOVE']:
        embedding = Embedding(output_dim=options['EMBEDDING_DIM'], weights = [options['EMBEDDING_MATRIX']] ,input_dim = len(options['VOCAB']) + 2, mask_zero=True ) 
    else:
        embedding = Embedding(output_dim=options['EMBEDDING_DIM'], input_dim = len(options['VOCAB']) + 2, mask_zero=True ) 
    ctxt_embed = embedding(ctxt)
    gold_resp_embed = embedding(gold_resp)
    alt_resp_embed = embedding(alt_resp)

    ctxt_gru = Bidirectional(GRU(options['CTXT_GRU_HIDDEN_STATE'], return_sequences=True))
    encoded_ctxt = ctxt_gru(ctxt_embed)

    shared_gru = Bidirectional(GRU(options['RESP_GRU_HIDDEN_STATE']))
    encoded_gold_resp = shared_gru(gold_resp_embed)
    encoded_alt_resp = shared_gru(alt_resp_embed)
    attention_module = AttentionLayer(options['CTXT_GRU_HIDDEN_STATE'], bilinear = options['BILINEAR'], return_att = True) 
    attended_gold_ctxt, gold_alpha = attention_module( [encoded_ctxt, encoded_gold_resp] )
    attended_alt_ctxt, alt_alpha = attention_module( [encoded_ctxt, encoded_alt_resp] )

    merged_vector = keras.layers.concatenate([attended_gold_ctxt, attended_alt_ctxt, encoded_gold_resp, encoded_alt_resp], axis = -1)
    if options['DROPOUT'] > 0.:
        merged_vector = Dropout(options['DROPOUT'])(merged_vector)
    merged_vector = Dense(options['DENSE_HIDDEN_STATE'], activation='tanh')(merged_vector)

    if options['DROPOUT'] > 0.:
        merged_vector = Dropout(options['DROPOUT'])(merged_vector)
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    model = Model(inputs=[ctxt, gold_resp, alt_resp], outputs=predictions)
    adam = optimizers.Adam(clipnorm=1.)
    model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model



if __name__ == "__main__":
    args = get_arguments()
    options = get_options(args)             
    if options['USE_GLOVE']:
        options['EMBEDDING_MATRIX'] = np.load(open(options['EMBED_PATH'],'r'))
    model = create_model(options)
    if options['LOAD_WEIGHTS']:
        model = get_best_model(options['SAVE_PREFIX'], model)

    filepath = options['SAVE_PREFIX'] + '_{epoch:02d}_{val_acc:.4f}.weights'
    weight_save = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
              patience=5, min_lr=options['MIN_LR'])
    print 'LOADING TRAINING DATA FROM: ',options['TRAIN']
    sys.stdout.flush()
    train_X, train_Y = generate_X_Y(options, 'TRAIN')
    train_gen = data_generator(train_X, train_Y, options)
    print 'LOADING VALIDATION DATA FROM: ',options['VAL']
    sys.stdout.flush()
    val_X, val_Y = generate_X_Y(options,'VAL', num_elem = 50000)
    val_gen = data_generator(val_X, val_Y, options)
    print 'STARTING TRAINING ....'
    #steps_epoch = len(train_X) / options['BATCH_SIZE'] if len(train_X) % options['BATCH_SIZE'] == 0else (len(train_X) / options['BATCH_SIZE']) + 1
    steps_val = len(val_X) / options['BATCH_SIZE'] if len(val_X) % options['BATCH_SIZE'] == 0  else (len(val_X) / options['BATCH_SIZE']) + 1    
    steps_epoch = 500
    #print steps_epoch
    #print steps_val
    model.fit_generator(train_gen, steps_per_epoch=steps_epoch, epochs=500, 
              validation_data=val_gen, validation_steps=steps_val,
              callbacks=[weight_save, reduce_lr])   
