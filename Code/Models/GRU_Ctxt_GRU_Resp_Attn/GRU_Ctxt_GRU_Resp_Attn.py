import keras
from keras.layers import Input, GRU, Embedding, Dense, Activation, add, RepeatVector, TimeDistributed, Flatten
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *
from keras import backend as K
from keras.engine.topology import Layer
import theano
import theano.tensor as T

from sklearn.externals import joblib

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pdb

# model params
CTXT_GRU_HIDDEN_STATE = 512
RESP_GRU_HIDDEN_STATE = 300
VOCAB_SIZE = 50000    # 1 for UNK
MAX_RESP_LEN = 50
MAX_CTX_LEN = 500
EMBEDDING_DIM = 300
DENSE_HIDDEN_STATE = 30

# training details
TRAIN_SIZE = 10000    # -1 => train on all
BATCH_SIZE = 256

# hpc file paths
train_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/train.txt'
val_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/val.txt'
test_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/test.txt'
count_vect_vocab_file = '/home/cse/dual/cs5130275/DERP/Code/Models/LogisticRegBaseline/vocab_50k'
save_model_path = '/scratch/cse/dual/cs5130275/DERP/Models/GRU_Ctxt_GRU_Resp_Attn/' + 'CTXT_HIDDEN_STATE_' + str(CTXT_GRU_HIDDEN_STATE) + 'RESP_HIDDEN_STATE_' + str(RESP_GRU_HIDDEN_STATE) + '_VOCAB_SIZE_' + str(VOCAB_SIZE) + '_MAX_RESP_LEN_' + str(MAX_RESP_LEN) + '_EMBEDDING_DIM_' + str(EMBEDDING_DIM) + '_DENSE_HIDDEN_STATE_' + str(DENSE_HIDDEN_STATE) + '_BATCH_SIZE_' + str(BATCH_SIZE)
load_model_path = ''

# local file paths
# train_file = 'train100.txt'
# count_vect_vocab_file = '../LogisticRegBaseline/vocab_50k'
# tfidf_transformer_file = '../LogisticRegBaseline/tfidf_transformer_50k'

class WeightSave(Callback):
    def setModelFile(self, model_file):
        self.model_file = model_file

    def on_train_begin(self, logs={}):
        if self.load_model_path:
            print('LOADING WEIGHTS FROM : ' + self.load_model_path)
            weights = joblib.load(self.load_model_path)
            self.model.set_weights(weights)

    def on_epoch_end(self, epochs, logs={}):
        cur_weights = self.model.get_weights()
        joblib.dump(cur_weights, self.model_file + '_on_epoch_' + str(epochs) + '.weights')

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

def data_generator(data_x, data_y, vocab_dict, inv_vocab):
    assert(0 not in inv_vocab)  # 0 is for masking
    
    while True:
        for i in range(0, len(data_x), BATCH_SIZE):
            cur_batch_x = data_x[i:i+BATCH_SIZE]
            cur_batch_x = [[y.lower() for y in x] for x in cur_batch_x]  # !when making vocab all are lower case
            cur_batch_y = data_y[i:i+BATCH_SIZE]
            
            cur_batch_x = [[TreebankWordTokenizer().tokenize(x[0]),TreebankWordTokenizer().tokenize(x[1]),TreebankWordTokenizer().tokenize(x[2])] for x in cur_batch_x]
            cur_batch_y = [y for i,y in enumerate(cur_batch_y) if len(cur_batch_x[i][0]) <= MAX_CTX_LEN and len(cur_batch_x[i][1])<=MAX_RESP_LEN and len(cur_batch_x[i][2])<=MAX_RESP_LEN]
            cur_batch_x = [x for x in cur_batch_x if len(x[0]) <= MAX_CTX_LEN and len(x[1])<=MAX_RESP_LEN and len(x[2])<=MAX_RESP_LEN]
            cur_batch_ctxt, cur_batch_gold_resp, cur_batch_alt_resp = zip(*cur_batch_x)
            
            # indices for context, 0 if nothing
            cur_batch_ctxt_vec = np.zeros((len(cur_batch_ctxt), MAX_CTX_LEN))

            # indices for responses, 0 if nothing
            cur_batch_gold_resp_vec = np.zeros((len(cur_batch_gold_resp),MAX_RESP_LEN))
            cur_batch_alt_resp_vec = np.zeros((len(cur_batch_gold_resp),MAX_RESP_LEN))

            for j in range(len(cur_batch_gold_resp)):
                for k in range(len(cur_batch_ctxt[j])):
                    if cur_batch_ctxt[j][k] in vocab_dict:
                        cur_batch_ctxt_vec[j][k] = vocab_dict[cur_batch_ctxt[j][k]]
                    else:
                        cur_batch_ctxt_vec[j][k] = vocab_dict['UNK']

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
            
            yield [cur_batch_ctxt_vec, cur_batch_gold_resp_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)

class SingleAttentionLayer(Layer):
    def __init__(self, dense_dim, return_att = False, **kwargs):
        self.return_att = return_att
        self.dense_dim = dense_dim
        super(SingleAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ctxt_dim = input_shape[0][2]
        self.resp_dim = input_shape[1][1]
        self.ctxt_dense = self.add_weight(shape=(self.ctxt_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.resp_dense = self.add_weight(shape=(self.resp_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.alpha_dense = self.add_weight(shape=(self.dense_dim, 1), initializer='glorot_uniform', trainable=True) # n_dim x 1
        super(SingleAttentionLayer, self).build(input_shape)

    def call(self, input_tensors, mask = None):
        ''' wbw attention layer:
        :param ctxt (input_tensors[0]) : batch_size x T x ctxt_dim
        :param resp (input_tensors[1]) : batch_size x resp_dim
        '''

        ctxt = input_tensors[0]
        resp = input_tensors[1]
        ctxt_mask = mask[0]

        resp_w = T.dot(resp, self.resp_dense) # bt_sz x dense_dim
        ctxt_w = T.dot(ctxt, self.ctxt_dense) # bt_sz x T x dense_dim
        resp_w_rep = resp_w[:,None,:] # bt_sz x T x dense_dim
        pre_alpha = T.tanh(ctxt_w + resp_w_rep) # bt_sz x T x dense_dim
        unnorm_alpha = T.dot(pre_alpha, self.alpha_dense).flatten(2) # bt_sz x T
        if ctxt_mask:
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

# def single_attention(ctxt, resp, ctxt_dense, resp_dense, alpha_dense):
#     pre_alpha = Activation('tanh')(add([RepeatVector(MAX_CTX_LEN)(resp_dense(resp)), TimeDistributed(ctxt_dense)(ctxt)]))
#     alpha = Activation("softmax")(Flatten()(TimeDistributed(alpha_dense)(pre_alpha)))
#     alpha = Reshape((MAX_CTX_LEN, 1), input_shape=(MAX_CTX_LEN,))(alpha)
#     permuted_ctxt = K.permute_dimensions(ctxt, (0,2,1))
#     return K.T.batched_dot(permuted_ctxt, alpha)

def create_model():
    ctxt = Input(shape=(MAX_CTX_LEN,))
    gold_resp = Input(shape=(MAX_RESP_LEN,))
    alt_resp = Input(shape=(MAX_RESP_LEN,))
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE+2, mask_zero=True)   # +1 for 'UNK', +1 for mask (0 can't be used)
    
    ctxt_emb = embedding(ctxt)
    gold_resp_emb = embedding(gold_resp)
    alt_resp_emb = embedding(alt_resp)

    ctxt_gru = Bidirectional(GRU(CTXT_GRU_HIDDEN_STATE, return_sequences = True))
    encoded_ctxt = ctxt_gru(ctxt_emb)

    shared_gru = Bidirectional(GRU(RESP_GRU_HIDDEN_STATE))
    encoded_gold_resp = shared_gru(gold_resp_emb)
    encoded_alt_resp = shared_gru(alt_resp_emb)

    # shared_ctxt_dense = Dense(CTXT_GRU_HIDDEN_STATE)
    # shared_resp_dense = Dense(CTXT_GRU_HIDDEN_STATE)
    # shared_alpha_dense = Dense(1)
    # attended_gold_ctxt = single_attention(encoded_ctxt, encoded_gold_resp, shared_ctxt_dense, shared_resp_dense, shared_alpha_dense)
    # attended_alt_ctxt = single_attention(encoded_ctxt, encoded_alt_resp, shared_ctxt_dense, shared_resp_dense, shared_alpha_dense)

    attention_module = SingleAttentionLayer(CTXT_GRU_HIDDEN_STATE, return_att = True)
    attended_gold_ctxt, gold_alpha = attention_module([encoded_ctxt, encoded_gold_resp])
    attended_alt_ctxt, ctxt_alpha = attention_module([encoded_ctxt, encoded_alt_resp])

    merged_vector = keras.layers.concatenate([attended_gold_ctxt, attended_alt_ctxt, encoded_gold_resp, encoded_alt_resp], axis=-1)
    merged_vector = Dense(DENSE_HIDDEN_STATE, activation='tanh')(merged_vector)
    
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model(inputs=[ctxt, gold_resp, alt_resp], outputs=predictions)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model

if __name__=='__main__':
    
    train_x, train_y = load_data(train_file, TRAIN_SIZE)
    val_x, val_y = load_data(val_file, 10000)
    count_vect_vocab = joblib.load(count_vect_vocab_file)
    # tfidf_transformer = joblib.load(tfidf_transformer_file)
    
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    # count_vect = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
    # count_vect.vocabulary_ = count_vect_vocab
    vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}
    
    train_gen = data_generator(train_x, train_y, vocab_dict, inv_vocab)
    val_gen = data_generator(val_x, val_y, vocab_dict, inv_vocab)

    model = create_model()
    weight_save = WeightSave()
    weight_save.model_file = save_model_path
    weight_save.load_model_path = load_model_path

    model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10)
    
    # model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10, validation_data=val_gen, validation_steps=len(val_x)/BATCH_SIZE, callbacks=[weight_save])

