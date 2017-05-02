import keras
from keras.layers import Input, GRU, Embedding, Dense, Activation, add, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *
from keras.optimizers import *
from keras.initializers import Constant

from Recurrent_Decoder import AttentionDecoderGRU

from configurations import *
from utils import *

from sklearn.externals import joblib
import numpy as np
import pdb

import sys
import cPickle

def create_model():
    ctxt = Input(shape=(MAX_CTX_LEN,))
    gold_resp = Input(shape=(MAX_RESP_LEN,))
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE+2, mask_zero=True)   # +1 for 'UNK', +1 for mask (0 can't be used)
    
    ctxt_emb = embedding(ctxt)
    gold_resp_emb = embedding(gold_resp)

    ctxt_gru = Bidirectional(GRU(CTXT_GRU_HIDDEN_STATE, return_sequences = True))
    encoded_ctxt = ctxt_gru(ctxt_emb)

    decoder_gru = AttentionDecoderGRU(RESP_GRU_HIDDEN_STATE, ATTN_DIM)
    decoder_resp_emb, attended_ctxt = AttentionDecoderGRU([gold_resp_emb, encoded_ctxt])

    merged_vector = keras.layers.concatenate([gold_resp_emb, decoder_resp_emb, attended_ctxt], axis=-1)
    # if DROPOUT > 0.0:
    # 	merged_vector = Dropout(DROPOUT)(merged_vector)
    merged_vector = Dense(DENSE_HIDDEN_STATE, activation='relu', bias_initializer=Constant(0.0005))(merged_vector)
    if DROPOUT > 0.0:
    	merged_vector = Dropout(DROPOUT)(merged_vector)
    
    decoder_resp = Dense(VOCAB_SIZE+2, activation='softmax')(merged_vector)
    
    model = Model(inputs=[ctxt, gold_resp], outputs=decoder_resp)
    adam = Adam(clipnorm=1.)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()

    return model

if __name__=='__main__':

    # count_vect_vocab = joblib.load(count_vect_vocab_file)
    glove_embedding = np.load(glove_embedding_file)
    # tfidf_transformer = joblib.load(tfidf_transformer_file)
    
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    # count_vect = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
    # count_vect.vocabulary_ = count_vect_vocab
    # vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}

    model = create_model()
    weight_save = WeightSave()
    weight_save.model_file = save_model_path
    weight_save.load_model_path = load_model_path
    
    else:
        train_x, train_y = load_data(train_file, TRAIN_SIZE)
        val_x, val_y = load_data(val_file, 10000)

        train_gen = data_generator(train_x, train_y, vocab_dict)
        val_gen = data_generator(val_x, val_y, vocab_dict)

        # model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10)
        if not load_model_path: model.layers[3].set_weights([glove_embedding]) 
        model.fit_generator(train_gen, steps_per_epoch=500, epochs=500, validation_data=val_gen, validation_steps=len(val_x)/BATCH_SIZE, callbacks=[weight_save])