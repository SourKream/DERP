import keras
from keras.layers import Input, GRU, Embedding, Dense
from keras.models import Model
from keras.layers.wrappers import Bidirectional

from sklearn.externals import joblib

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pdb

# model params
GRU_HIDDEN_STATE = 300
VOCAB_SIZE = 50000    # 1 for UNK
MAX_RESP_LEN = 50
EMBEDDING_DIM = 100
DENSE_HIDDEN_STATE = 30

# training details
TRAIN_SIZE = 100    # -1 => train on all
BATCH_SIZE = 32

# hpc file paths
train_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/train.txt'
val_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/val.txt'
test_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/test.txt'
count_vect_vocab_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/LogisticRegBaseline/vocab_50k'
tfidf_transformer_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/LogisticRegBaseline/tfidf_transformer_50k'

# local file paths
# train_file = 'train100.txt'
# count_vect_vocab_file = 'vocab_50k'
# tfidf_transformer_file = 'tfidf_transformer_50k'

def load_data(filename, num_dat_points=-1):
    f = open(filename, 'rt')
    
    if num_dat_points==-1:
        dat = f.readlines()
    else:
        dat = []
        for i in range(num_dat_points):
            dat.append(f.readline().strip())
            
    f.close()
    
    dat = [x.split('\t') for x in dat]
    assert(all([len(x)==4 for x in dat]))
    dat_x = [x[:3] for x in dat]
    dat_y = [int(x[3]) for x in dat]
    
    return dat_x, dat_y

def data_generator(data_x, data_y, vocab_dict, inv_vocab, count_vect, tfidf_transformer):
    assert(0 not in inv_vocab)  # 0 is for masking
    
    while True:
        for i in range(0, len(data_x), BATCH_SIZE):
            cur_batch_x = data_x[i:i+BATCH_SIZE]
            cur_batch_x = [[y.lower() for y in x] for x in cur_batch_x]  # !when making vocab all are lower case
            cur_batch_y = data_y[i:i+BATCH_SIZE]
            
            cur_batch_x = [[x[0],TreebankWordTokenizer().tokenize(x[1]),TreebankWordTokenizer().tokenize(x[2])] for x in cur_batch_x]
            cur_batch_y = [y for i,y in enumerate(cur_batch_y) if len(cur_batch_x[i][1])<=MAX_RESP_LEN and len(cur_batch_x[i][2])<=MAX_RESP_LEN]
            cur_batch_x = [x for x in cur_batch_x if len(x[1])<=MAX_RESP_LEN and len(x[2])<=MAX_RESP_LEN]
            cur_batch_ctxt, cur_batch_gold_resp, cur_batch_alt_resp = zip(*cur_batch_x)
            
            # tfidf for context
            cur_batch_ctxt = count_vect.transform(cur_batch_ctxt)
            cur_batch_ctxt = tfidf_transformer.transform(cur_batch_ctxt)
            
            # indices for responses, 0 if nothing
            cur_batch_gold_resp_vec = np.zeros((len(cur_batch_gold_resp),MAX_RESP_LEN))
            cur_batch_alt_resp_vec = np.zeros((len(cur_batch_gold_resp),MAX_RESP_LEN))
            
            for j in range(len(cur_batch_gold_resp)):
                for k in range(len(cur_batch_gold_resp[j])):
                    if cur_batch_gold_resp[j][k] in vocab_dict:
                        cur_batch_gold_resp_vec[j][k] = vocab_dict[cur_batch_gold_resp[j][k]]
                    else:
                        cur_batch_gold_resp_vec[j][k] = vocab_dict['UNK']
            
                for k in range(len(cur_batch_alt_resp[j])):
                    if cur_batch_alt_resp[j][k] in vocab_dict:
                        cur_batch_alt_resp_vec[j][k] = vocab_dict[cur_batch_alt_resp[j][k]]
                    else:
                        cur_batch_alt_resp_vec[j][k] = vocab_dict['UNK']
            
            yield [cur_batch_ctxt.todense(), cur_batch_gold_resp_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)
        
    
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
    
    train_x, train_y = load_data(train_file, TRAIN_SIZE)
    count_vect_vocab = joblib.load(count_vect_vocab_file)
    tfidf_transformer = joblib.load(tfidf_transformer_file)
    
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    count_vect = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
    count_vect.vocabulary_ = count_vect_vocab
    vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}
    
    train_gen = data_generator(train_x, train_y, vocab_dict, inv_vocab, count_vect, tfidf_transformer)
    next(train_gen)
    model = create_model()
    
    model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10)