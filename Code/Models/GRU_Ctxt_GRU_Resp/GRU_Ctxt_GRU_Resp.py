import keras
from keras.layers import Input, GRU, Embedding, Dense, Dropout
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *

from configurations import *
from utils import *

from sklearn.externals import joblib
import numpy as np
import pdb

import sys
   
def create_model():
    ctxt = Input(shape=(MAX_CTX_LEN,))
    gold_resp = Input(shape=(MAX_RESP_LEN,))
    alt_resp = Input(shape=(MAX_RESP_LEN,))
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE+2, mask_zero=True)   # +1 for 'UNK', +1 for mask (0 can't be used)
    
    ctxt_emb = embedding(ctxt)
    gold_resp_emb = embedding(gold_resp)
    alt_resp_emb = embedding(alt_resp)

    ctxt_gru = Bidirectional(GRU(CTXT_GRU_HIDDEN_STATE))
    encoded_ctxt = ctxt_gru(ctxt_emb)

    shared_gru = Bidirectional(GRU(RESP_GRU_HIDDEN_STATE))
    encoded_gold_resp = shared_gru(gold_resp_emb)
    encoded_alt_resp = shared_gru(alt_resp_emb)
    
    merged_vector = keras.layers.concatenate([encoded_ctxt, encoded_gold_resp, encoded_alt_resp], axis=-1)
    if DROPOUT > 0.0:
    	merged_vector = Dropout(DROPOUT)(merged_vector)
    merged_vector = Dense(DENSE_HIDDEN_STATE, activation='tanh')(merged_vector)
    if DROPOUT > 0.0:
    	merged_vector = Dropout(DROPOUT)(merged_vector)

    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model(inputs=[ctxt, gold_resp, alt_resp], outputs=predictions)
    adam = Adam(clipnorm=1.)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model

if __name__=='__main__':

    count_vect_vocab = joblib.load(count_vect_vocab_file)
    # tfidf_transformer = joblib.load(tfidf_transformer_file)
    
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    # count_vect = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
    # count_vect.vocabulary_ = count_vect_vocab
    vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}

    model = create_model()
    weight_save = WeightSave()
    weight_save.model_file = save_model_path
    weight_save.load_model_path = load_model_path

    if len(sys.argv) > 1 and sys.argv[1] == 'Test':
        assert load_model_path != ''

        print 'Loading Data ...'
        test_x, test_y = load_data(test_file, -1)
        test_gen = data_generator(test_x, test_y, vocab_dict)
        print 'Restoring Weights ...'
        model.set_weights(joblib.load(load_model_path))

        print 'Testing ...'
        print model.evaluate_generator(test_gen, steps = len(test_x)/BATCH_SIZE)

    else:
        train_x, train_y = load_data(train_file, TRAIN_SIZE)
        val_x, val_y = load_data(val_file, 10000)

        train_gen = data_generator(train_x, train_y, vocab_dict)
        val_gen = data_generator(val_x, val_y, vocab_dict)

        # model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10)
    
        model.fit_generator(train_gen, steps_per_epoch=500, epochs=500, validation_data=val_gen, validation_steps=len(val_x)/BATCH_SIZE, callbacks=[weight_save])