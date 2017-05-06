from sklearn.externals import joblib
from keras.layers import Input, GRU, Embedding, Dense, Dropout 
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *
from keras import optimizers
# from configurations import *
# from utils import *
from CustomEmbeddingLayer import *
from keras.engine.topology import Layer
from sklearn.externals import joblib
import keras
import keras.backend as K
import numpy as np
import pdb
import theano.tensor as T
import sys

from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.legacy import interfaces

from keras import activations
from keras.engine import InputSpec
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import BatchNormalization
from keras.layers import *
 
###############--------------------- Configurations -------------- ###########
def get_options():
	options = {}
	
	options['TRAIN'] = True # False is testing
	options['LOAD_WEIGHTS'] = False # True then resumes training from a pre-exting model
	# Network configuration
	options['BATCH_SIZE'] = 256
	options['MAX_CTX_LEN'] = 10 
	options['MAX_RESP_LEN'] = 5 
	options['MAX_WORD_LEN'] = 5 # Set this as necessary
	options['DROPOUT'] = 0.0
	# Embedding layer details
	options['EMBEDDING_DIM'] = 30
	options['CHAR_LSTM_DIM'] = 30
	# Hidden state details
	options['CTXT_GRU_HIDDEN_STATE'] = 10
	options['RESP_GRU_HIDDEN_STATE'] = 10
	options['DENSE_HIDDEN_STATE'] = 20
	options['BILINEAR'] = False	
	# Now the vocabulary 
	options['DATA_DIR'] = '/scratch/cse/btech/cs1130773/DERPData/Data/'
	options['VOCAB_PATH'] = options['DATA_DIR'] + 'vocab_char'
	options['VOCAB'] = {'a':1, 'b':2}
	# options['VOCAB'] = joblib.load(options['VOCAB_PATH'])

	options['USE_GLOVE'] = False
	options['EMBED_PATH'] = options['DATA_DIR'] + 'glove_embedding_vocab50k.npy'
	
	# Now the data files
	options['TRAIN'] = options['DATA_DIR'] + 'train.txt'
	options['VAL'] =  options['DATA_DIR'] + 'val.txt'
	options['TEST'] = options['DATA_DIR'] + 'test.txt'

	# Now the model save path
	#TODO
	options['SAVE_DIR'] = "/scratch/cse/btech/cs1130773/DERPData/GRU_CTXT_GRU_RESP_ATTN_CHAR/" 
	options['SAVE_PREFIX'] = options['SAVE_DIR'] + "model_" 

	# if restrict the size of the validation set
	options['VAL_SIZE'] = 10

	return options
###############--------------------- Utilities ------------------- ###########
class WeightSave(Callback):
    def __init__(self, options):
        self.options = options
    def on_train_begin(self, logs={}):
        if self.options['LOAD_WEIGHTS']:
            assert 'MODEL_FILE' in self.options
            print('LOADING WEIGHTS FROM : ' + self.options['MODEL_FILE'])
            weights = joblib.load( self.options['MODEL_FILE'] )
            self.model.set_weights(weights)
    def on_epoch_end(self, epochs, logs = {}):
        cur_weights = self.model.get_weights()
        joblib.dump(cur_weights, self.options['SAVE_PREFIX'] + '_on_epoch_' + str(epochs) + '.weights')

def update_X(X, sentence, key,options ):
    sequence = sentence.split(' ') # TODO : A better tokenizer
    seq_len = options['MAX_CTX_LEN'] if key == 'CONTEXT' else options['MAX_RESP_LEN']
    sent = []    
    for idx in xrange(seq_len):
        if idx < len(sequence):
            word  = sequence[idx]
            word_idx = []
            for jdx in xrange(options['MAX_WORD_LEN']):
                if jdx < len(word):
                    char = word[jdx]
                    word_idx.append(options['VOCAB'][char] if char in options['VOCAB'] else len(options['VOCAB']))
                else:
                    word_idx.append(0)
            sent.append(word_idx)
        else:
            sent.append([0 for idx in xrange(options['MAX_WORD_LEN'])])

    X[key].append(sent)
def generate_X_Y_idx(options,key='TRAIN'):  
    X = {'CONTEXT':[],'GOLD':[],'RESP':[]}
    y = []
    with open(options[key],'rb') as f:
        for line in f:
            line = line.strip().split('\t')
            context = line[0]
            gold = line[1]
            altenate = line[2]
            update_X(X, context, 'CONTEXT',options)
            update_X(X, gold, 'GOLD',options)
            update_X(X, altenate, 'RESP',options)            
                        
            y.append(int(line[3]))
            if key == 'VAL' :
                if len(X['CONTEXT']) == options['VAL_SIZE']:
                    assert len(y) == options['VAL_SIZE']
                    break
    for key in X:
        X[key] = np.array(X[key])
    y = np.array(y)
    return X,y

def data_generator(data_x, data_y, options):
    BATCH_SIZE = options['BATCH_SIZE']
    MAX_RESP_LEN = options['MAX_RESP_LEN']
    MAX_CTX_LEN = options['MAX_CTX_LEN']
    MAX_WORD_LEN = options['MAX_WORD_LEN']
    while True:
        for i in xrange(0, len(data_x['CONTEXT']), BATCH_SIZE):
            cur_batch_x_ctxt = data_x['CONTEXT'][i:i+BATCH_SIZE]
            
            cur_batch_x_gold = data_x['GOLD'][i:i+BATCH_SIZE]

            cur_batch_x_resp = data_x['RESP'][i:i+BATCH_SIZE]

            cur_batch_y = data_y[i:i+BATCH_SIZE]                                
            
            yield [cur_batch_x_ctxt,cur_batch_x_gold,cur_batch_x_resp], np.array(cur_batch_y)

###############--------------------- Utilities ------------------- ###########
	
def get_H_i(i):  
    # get element i from time dimension
    def get_X_i(X):
        return X[:,i,:];
    return get_X_i

def CharaterLevelEmbeddingLayer(input_max_length, character_embedding_layer, character_LSTM_layer):

	data_input = Input(shape=(input_max_length, options['MAX_WORD_LEN'])) # bt_sz x T x W

	# character_embedding_layer = Embedding(len(options['VOCAB']) + 2, options['EMBEDDING_DIM'], input_length = options['MAX_WORD_LEN'], name="Char Embedding Layer")
	# character_LSTM_layer = Bidirectional(LSTM(options['CHAR_LSTM_DIM'], return_sequences = False, name="Char LSTM Layer"))
	expand_dim = Reshape((1,2*options['CHAR_LSTM_DIM']), input_shape=(2*options['CHAR_LSTM_DIM'],))

	word_encode = []
	for t in range(input_max_length):
		word_t = Lambda(get_H_i(t), output_shape=(options['MAX_WORD_LEN'],), name='Word_'+str(t))(data_input)
		word_t_embed = character_embedding_layer(word_t)
		word_t_encode = character_LSTM_layer(word_t_embed)
		word_t_encode = expand_dim(word_t_encode)
		word_encode.append(word_t_encode)
	word_encoded_input = concatenate(word_encode, axis=1) # bt_sz x T x k

	model = Model(inputs=data_input, outputs=word_encoded_input)
	return model

def create_model(options):

	ctxt = Input(shape=(options['MAX_CTX_LEN'],options['MAX_WORD_LEN'])) # bt_sz x T x W
	gold_resp = Input(shape=(options['MAX_RESP_LEN'],options['MAX_WORD_LEN'])) # bt_sz x T x W
	alt_resp = Input(shape=(options['MAX_RESP_LEN'],options['MAX_WORD_LEN'])) # bt_sz x T x W
	
	## Embedding 
	character_embedding_layer = Embedding(len(options['VOCAB']) + 2, options['EMBEDDING_DIM'], input_length = options['MAX_WORD_LEN'], name="Char Embedding Layer")
	character_LSTM_layer = Bidirectional(LSTM(options['CHAR_LSTM_DIM'], return_sequences = False, name="Char LSTM Layer"))

	ctxt_embed = CharaterLevelEmbeddingLayer(options['MAX_CTX_LEN'], character_embedding_layer, character_LSTM_layer)(ctxt)	# bt_sz x T x n_embed
	gold_resp_embed = CharaterLevelEmbeddingLayer(options['MAX_RESP_LEN'], character_embedding_layer, character_LSTM_layer)(gold_resp)	# bt_sz x T x n_embed
 	alt_resp_embed = CharaterLevelEmbeddingLayer(options['MAX_RESP_LEN'], character_embedding_layer, character_LSTM_layer)(alt_resp)	# bt_sz x T x n_embed


	ctxt_gru = Bidirectional(GRU(options['CTXT_GRU_HIDDEN_STATE'], return_sequences=False))
	encoded_ctxt = ctxt_gru(ctxt_embed) # bt_sz x n_dim

	shared_gru = Bidirectional(GRU(options['RESP_GRU_HIDDEN_STATE']))
	encoded_gold_resp = shared_gru(gold_resp_embed) # bt_sz x n_dim
	encoded_alt_resp = shared_gru(alt_resp_embed) # bt_sz x n_dim
	
	# attention_module = AttentionLayer(options['CTXT_GRU_HIDDEN_STATE'], bilinear = options['BILINEAR'], return_att = True) 
	# attended_gold_ctxt, gold_alpha = attention_module( [encoded_ctxt, encoded_gold_resp] )
	# attended_alt_ctxt, alt_alpha = attention_module( [encoded_ctxt, encoded_alt_resp] )

	# merged_vector = keras.layers.concatenate([attended_gold_ctxt, attended_alt_ctxt, encoded_gold_resp, encoded_alt_resp], axis = -1)
	merged_vector = keras.layers.concatenate([encoded_ctxt, encoded_gold_resp, encoded_alt_resp], axis = -1)
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
	options = get_options()	
	if options['LOAD_WEIGHTS']:
		epoch = 10
		options['MODEL_FILE'] = options['SAVE_PREFIX'] + '_on_epoch_' + str(epoch) + '.weights'
	if options['USE_GLOVE']:
		options['EMBEDDING_MATRIX'] = np.load(open(options['EMBED_PATH'],'r'))

	model = create_model(options)
	exit(0)
	weight_save = WeightSave(options)	

	if options['TRAIN']:
		print 'GETTING TRAINING DATA NOW FROM FILE ... ', options['TRAIN']	
		train_X, train_Y = generate_X_Y_idx(options, 'TRAIN')
		train_gen = data_generator(train_X, train_Y, options)
		print 'GETTING VALIDATION DATA NOW...', options['VAL']
		val_X, val_Y = generate_X_Y_idx(options,'VAL')		
		val_gen = data_generator(val_X, val_Y, options)
		print 'STARTING TRAINING NOW ...'
		#steps_epoch = len(train_X) / options['BATCH_SIZE'] if len(train_X) % options['BATCH_SIZE'] == 0else (len(train_X) / options['BATCH_SIZE']) + 1
		steps_val = len(val_X) / options['BATCH_SIZE'] if len(val_X) % options['BATCH_SIZE'] == 0  else (len(val_X) / options['BATCH_SIZE']) + 1
		steps_epoch = 500
		#print steps_epoch
		#print steps_val
		model.fit_generator(train_gen, steps_per_epoch=steps_epoch, epochs=500, 
				  validation_data=val_gen, validation_steps=steps_val,
				  callbacks=[weight_save])
	else:
		#TODO Fix this
		pass
