import keras
from keras.layers import Input, GRU, Embedding, Dense
from keras.models import Model
from keras.callbacks import Callback
from keras.layers.wrappers import Bidirectional
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from my_utils import *
import numpy as np
import pdb

# model params
GRU_HIDDEN_STATE = 300
VOCAB_SIZE = 50000    # 1 for UNK
MAX_RESP_LEN = 50
EMBEDDING_DIM = 300
DENSE_HIDDEN_STATE = 30

# training details
TRAIN_SIZE = -1    # -1 => train on all
BATCH_SIZE = 256

# hpc file paths
train_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/train.txt'
val_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/val.txt'
test_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/test.txt'
count_vect_vocab_file = '/home/cse/dual/cs5130275/DERP/Code/Models/LogisticRegBaseline/vocab_50k'
tfidf_transformer_file = '/home/cse/dual/cs5130275/DERP/Code/Models/LogisticRegBaseline/tfidf_transformer_50k'
save_model_path = '/scratch/cse/dual/cs5130275/DERP/Models/tfidf_Ctxt+GRU_response/' + 'GRU_HIDDEN_STATE_' + str(GRU_HIDDEN_STATE) + '_VOCAB_SIZE_' + str(VOCAB_SIZE) + '_MAX_RESP_LEN_' + str(MAX_RESP_LEN) + '_EMBEDDING_DIM_' + str(EMBEDDING_DIM) + '_DENSE_HIDDEN_STATE_' + str(DENSE_HIDDEN_STATE) + '_BATCH_SIZE_' + str(BATCH_SIZE)
load_model_path = ''

# local file paths
# train_file = 'train100.txt'
# count_vect_vocab_file = '../LogisticRegBaseline/vocab_50k'
# tfidf_transformer_file = '../LogisticRegBaseline/tfidf_transformer_50k'

class WeightSave(Callback):
    def setModelFile(self, model_file):
        self.model_file = model_file
    def on_train_begin(self, logs={}):
        if load_model_path:
            print('LOADING WEIGHTS FROM : ' + load_model_path)
            weights = joblib.load(load_model_path)
            self.model.set_weights(weights)
    def on_epoch_end(self, epochs, logs={}):
        cur_weights = self.model.get_weights()
        joblib.dump(cur_weights, self.model_file + '_on_epoch_' + str(epochs) + '.weights')

def create_model():
    ctxt_tfidf = Input(shape=(VOCAB_SIZE,))
    gold_resp = Input(shape=(MAX_RESP_LEN,))
    alt_resp = Input(shape=(MAX_RESP_LEN,))
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE+2, input_length=MAX_RESP_LEN, mask_zero=True)   # +1 for 'UNK', +1 for mask (0 can't be used)
    
    gold_resp_emb = embedding(gold_resp)
    alt_resp_emb = embedding(alt_resp)
    
    shared_gru = Bidirectional(GRU(GRU_HIDDEN_STATE))
    
    encoded_gold_resp = shared_gru(gold_resp_emb)
    encoded_alt_resp = shared_gru(alt_resp_emb)
    
    merged_vector = keras.layers.concatenate([ctxt_tfidf, encoded_gold_resp, encoded_alt_resp], axis=-1)
    merged_vector = Dense(DENSE_HIDDEN_STATE, activation='tanh')(merged_vector)
    
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model(inputs=[ctxt_tfidf, gold_resp, alt_resp], outputs=predictions)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model
    
if __name__=='__main__':
    
    # loads
    train_x, train_y = load_data(train_file, TRAIN_SIZE)
    val_x, val_y = load_data(val_file, 10000)
    count_vect_vocab = joblib.load(count_vect_vocab_file)
    tfidf_transformer = joblib.load(tfidf_transformer_file)
    print('loaded data!')
    
    # prepare vocab, count_vect
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    count_vect = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
    count_vect.vocabulary_ = count_vect_vocab
    vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}
    
    # generators
    train_gen = data_generator(train_x, train_y, vocab_dict, inv_vocab, count_vect, tfidf_transformer)
    val_gen = data_generator(val_x, val_y, vocab_dict, inv_vocab, count_vect, tfidf_transformer)
    
    # model/callbacks
    model = create_model()
    weight_save = WeightSave()
    weight_save.setModelFile(save_model_path)
    
    # train!
    model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10, validation_data=val_gen, validation_steps=len(val_x)/BATCH_SIZE, callbacks=[weight_save]) 