import keras
from keras.layers import Input, GRU, Embedding, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from my_utils import *
from config import *
import cPickle as cp
import bz2,json,contextlib
import numpy as np
import pdb


def create_model():
    ctxt_tfidf = Input(shape=(VOCAB_SIZE,))
    gold_resp = Input(shape=(MAX_RESP_LEN,))
    alt_resp = Input(shape=(MAX_RESP_LEN,))
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE+2, input_length=MAX_RESP_LEN, mask_zero=True)   # +1 for 'UNK', +1 for mask (0 can't be used)
    
    gold_resp_emb = embedding(gold_resp)
    alt_resp_emb = embedding(alt_resp)
    
    shared_gru = Bidirectional(GRU(GRU_HIDDEN_STATE, dropout=DROPOUT))
    
    encoded_gold_resp = shared_gru(gold_resp_emb)
    encoded_alt_resp = shared_gru(alt_resp_emb)
    
    merged_vector = keras.layers.concatenate([ctxt_tfidf, encoded_gold_resp, encoded_alt_resp], axis=-1)
    merged_vector = Dropout(DROPOUT)(merged_vector)
    merged_vector = Dense(DENSE_HIDDEN_STATE1, activation='relu')(merged_vector)
    merged_vector = Dropout(DROPOUT)(merged_vector)
    merged_vector = Dense(DENSE_HIDDEN_STATE2, activation='tanh')(merged_vector)
    merged_vector = Dropout(DROPOUT)(merged_vector)
        
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model(inputs=[ctxt_tfidf, gold_resp, alt_resp], outputs=predictions)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model
    
if __name__=='__main__':
    
    # loads
    train_x, train_y = load_data_raw(train_file, TRAIN_SIZE)
    val_x, val_y = load_data_raw(val_file, 50000)
    count_vect_vocab = joblib.load(count_vect_vocab_file)
    tfidf_transformer = joblib.load(tfidf_transformer_file)
    
    # preprocessed data loads
    with open(train_ctxt_tfidfed_file,'r') as f:
        train_ctxt_tfidfed = cp.load(f)
    with contextlib.closing(bz2.BZ2File(train_gold_resp_preprocessed_file, 'rb')) as f:
        train_gold_resp_preprocessed = json.load(f)
    with contextlib.closing(bz2.BZ2File(train_alt_resp_preprocessed_file, 'rb')) as f:
        train_alt_resp_preprocessed = json.load(f)
    
    print('loaded data!')
    
    # prepare vocab, count_vect
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    count_vect = CountVectorizer(tokenizer=my_tokenize)
    count_vect.vocabulary_ = count_vect_vocab
    vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}
    
    # generators
    # train_gen = data_generator_raw(train_x, train_y, vocab_dict, count_vect, tfidf_transformer)
    train_gen = data_generator_preprocessed(train_ctxt_tfidfed, train_gold_resp_preprocessed, train_alt_resp_preprocessed, train_y)
    val_gen = data_generator_raw(val_x, val_y, vocab_dict, count_vect, tfidf_transformer)
    
    # model/callbacks
    model = create_model()
    checkpointer = ModelCheckpoint(filepath=save_model_path, verbose=1, save_best_only=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'Test':
        test_routine(model)
        
    else:    
        # train!
        print('\n==> NOW TRAINING ' + model_name +'\n')
        model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE/10, epochs=10*10, validation_data=val_gen, validation_steps=len(val_x)/BATCH_SIZE, callbacks=[checkpointer]) 