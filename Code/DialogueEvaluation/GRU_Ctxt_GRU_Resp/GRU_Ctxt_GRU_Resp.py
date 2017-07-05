from sklearn.externals import joblib
import keras
from keras.layers import Input, GRU, Embedding, Dense, Dropout, concatenate
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

from utils import *
from keras.optimizers import Adam 
from sklearn.externals import joblib
import numpy as np
import pdb

import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')
from base_utils import ModelCheckpoint

'''
    This file is just for training the model. For predictions, use the predictions.py file
'''
def get_arguments():
    parser = argparse.ArgumentParser(description='No attention model')  
    parser.add_argument('-resume_training', action="store", default="False", dest="resume_training", type=str)
    opts = parser.parse_args(sys.argv[1:])      
    opts.resume_training = True if opts.resume_training == "True" else False
    return opts

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

    ctxt_gru = Bidirectional(GRU(options['CTXT_GRU_HIDDEN_STATE']))
    encoded_ctxt = ctxt_gru(ctxt_embed)

    shared_gru = Bidirectional(GRU(options['RESP_GRU_HIDDEN_STATE']))
    encoded_gold_resp = shared_gru(gold_resp_embed)
    encoded_alt_resp = shared_gru(alt_resp_embed)

    merged_vector = concatenate([encoded_ctxt, encoded_gold_resp, encoded_alt_resp], axis = -1)
    if options['DROPOUT'] > 0.:
        merged_vector = Dropout(options['DROPOUT'])(merged_vector)
    merged_vector = Dense(options['DENSE_HIDDEN_STATE'], activation='tanh')(merged_vector)

    if options['DROPOUT'] > 0.:
        merged_vector = Dropout(options['DROPOUT'])(merged_vector)
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    model = Model(inputs=[ctxt, gold_resp, alt_resp], outputs=predictions)
    adam = Adam(clipnorm=1.)
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

    filepath = options['SAVE_PREFIX'] + '_{epoch:02d}_{val_acc:.4f}.hdf5'
    weight_save = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
              patience=5, min_lr=options['MIN_LR'])
    print 'LOADING TRAINING DATA FROM: ',options['TRAIN']
    sys.stdout.flush()
    train_X, train_Y = generate_X_Y(options, 'TRAIN')
    print 'LOADING VALIDATION DATA FROM: ',options['VAL']
    sys.stdout.flush()
    train_gen = data_generator(train_X, train_Y, options)

    val_X, val_Y = generate_X_Y(options,'VAL', num_elem = 50000)        
    val_gen = data_generator(val_X, val_Y, options)
    val_steps = len(val_X) / options['BATCH_SIZE'] if len(val_X) % options['BATCH_SIZE'] == 0 else (len(val_X) / options['BATCH_SIZE']) + 1
    print 'STARTING TRAINING ....'
    sys.stdout.flush()
    model.fit_generator(train_gen, steps_per_epoch=500, epochs=500, 
              validation_data=val_gen, validation_steps= val_steps,
              callbacks=[weight_save, reduce_lr])
                