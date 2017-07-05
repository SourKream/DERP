from sklearn.externals import joblib
import keras
from keras.layers import Input, GRU, Embedding, Dense, Dropout 
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from keras import optimizers
from utils import *
import cPickle as cp
from keras.engine.topology import Layer
from sklearn.externals import joblib
import numpy as np
import pdb
import theano.tensor as T
import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')
from base_utils import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import spearmanr
from configurations import *
import argparse

def get_arguments():
	parser = argparse.ArgumentParser(description='Ruber Derp model')
	parser.add_argument('-margin', action="store", default=0.5, dest="margin", type=float)
	parser.add_argument('-vanilla', action="store", default='True', dest="vanilla", type=str)
	parser.add_argument('-bad_grammar', action="store", default='False', dest="bad_grammar", type=str)	
	opts = parser.parse_args(sys.argv[1:])	
	opts.vanilla = True if opts.vanilla == 'True' else False
	opts.bad_grammar = True if opts.bad_grammar == 'True' else False
	return opts


def max_margin_loss(y_true,y_pred):
	return K.mean( K.maximum(0., y_true - y_pred) ) # MARGIN - (y_pos - y_neg)
	#return K.mean( K.maximum(0., y_true[0] - y_pred[0] + y_pred[1]) )

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
			self.ctxt_dense = self.add_weight(name = None, shape=(self.ctxt_dim, self.resp_dim), initializer='glorot_uniform', trainable=True)
		else:
			self.ctxt_dense = self.add_weight(name = None, shape=(self.ctxt_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
			self.resp_dense = self.add_weight(name = None, shape=(self.resp_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
			self.alpha_dense = self.add_weight(name = None, shape=(self.dense_dim, 1), initializer='glorot_uniform', trainable=True) # n_dim x 1
		super(AttentionLayer, self).build(input_shape)

	def call(self, input_tensors, mask = None):
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
	
class BilinearLayer(Layer):
	def __init__(self, dense_dim, **kwargs):
		self.dense_dim = dense_dim
		super(BilinearLayer,self).__init__(**kwargs)
	
	def build(self, input_shapes):
		self.ctx_dim = input_shapes[0][1]
		self.resp_dim = input_shapes[1][1]
		self.ctx_dense = self.add_weight(name = None, shape=(self.ctx_dim, self.resp_dim), initializer='glorot_uniform', trainable=True)
	
	def call(self, input_tensors, mask=None):
		ctx = input_tensors[0]
		resp = input_tensors[1]
		M = self.ctx_dense
		output = T.batched_dot(T.dot(ctx,M), resp).dimshuffle((0,'x'))
		return output
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0][0],1)
	
	def compute_mask(self, input_tensors,input_masks):
		return None

class SubtractLayer(Layer):
	def __init__(self, **kwargs):
		super(SubtractLayer, self).__init__(**kwargs)
	
	def build(self, input_shapes):
		self.n_dim = input_shapes[0][1]
		self.bt_sz = input_shapes[0][0]
		assert input_shapes[0] == input_shapes[1]
	
	def call(self, input_tensors, mask=None):
		return input_tensors[0] - input_tensors[1]

	def compute_output_shape(self, input_shapes):
		return (input_shapes[0][0], self.n_dim)
	
	def compute_mask(self, input_tensors, input_masks):
		return None

def base_model(options):
	embedding = None
	ctxt = Input(shape=(options['MAX_CTX_LEN'],))
	resp = Input(shape=(options['MAX_RESP_LEN'],))
	if options['USE_GLOVE']:
		embedding = Embedding(output_dim=options['EMBEDDING_DIM'], weights = [options['EMBEDDING_MATRIX']] ,input_dim = len(options['VOCAB']) + 2, mask_zero=True ) 
	else:
		embedding = Embedding(output_dim=options['EMBEDDING_DIM'], input_dim = len(options['VOCAB']) + 2, mask_zero=True )
	ctxt_embed = embedding(ctxt)
	resp_embed = embedding(resp)
	ctxt_gru = Bidirectional(GRU(options['CTXT_GRU_HIDDEN_STATE'], return_sequences=True))
	encoded_ctxt = ctxt_gru(ctxt_embed)
	resp_gru = Bidirectional(GRU(options['RESP_GRU_HIDDEN_STATE']))
	encoded_resp = resp_gru(resp_embed)
	
	attention_module = AttentionLayer(options['CTXT_GRU_HIDDEN_STATE'], bilinear = options['BILINEAR'], return_att = True)
	attended_resp_ctxt, resp_alpha = attention_module( [encoded_ctxt, encoded_resp]  )
	if not options['VANILLA']:
		bilinear_module = BilinearLayer(options['CTXT_GRU_HIDDEN_STATE']) 
		bilinear_ctxt_resp = bilinear_module([attended_resp_ctxt, encoded_resp])
		merged_vec = keras.layers.concatenate([attended_resp_ctxt, encoded_resp, bilinear_ctxt_resp], axis=-1)
	else:
		merged_vec = keras.layers.concatenate([attended_resp_ctxt, encoded_resp], axis=-1)

	dense_vec = Dense(options['DENSE_HIDDEN_STATE'], activation='tanh')(merged_vec)
	pred_vec = Dense(1, activation='sigmoid')(dense_vec)
	return Model([ctxt, resp], pred_vec)

def create_model(options):	
	ctxt = Input(shape=(options['MAX_CTX_LEN'],))
	pos_resp = Input(shape=(options['MAX_RESP_LEN'],))
	neg_resp = Input(shape=(options['MAX_RESP_LEN'],))
	scoring_model = base_model(options)
	pos_score = scoring_model([ctxt, pos_resp])
	neg_score = scoring_model([ctxt, neg_resp])
	score = SubtractLayer()([pos_score, neg_score]) # pos_score - neg_score
	model = Model(inputs=[ctxt, pos_resp, neg_resp], outputs= score)
	adam = optimizers.Adam(clipnorm=1.)
	model.compile(optimizer=adam, loss=max_margin_loss,metrics=[max_margin_loss])
	model.summary()
	return model

if __name__ == "__main__":
	args = get_arguments()
	options = get_options(args)	
	print 'CONFIGURATIONS ...'
	print 'MARGIN ', options['MARGIN']
	if options['VANILLA']:
		print 'MODEL','VANILLA'
	else:
		print 'MODEL','BI-MATCHING'
	if args.bad_grammar:
		print 'MODEL', "considers bad grammar"
	else:
		print 'MODEL', "doesn't consider bad grammar"
	sys.stdout.flush()	
	if options['USE_GLOVE']:
		options['EMBEDDING_MATRIX'] = np.load(open(options['EMBED_PATH'],'r'))
	model = create_model(options)
	if options['LOAD_WEIGHTS']:
		model = get_best_model(options['SAVE_PREFIX'], model)		
	filepath = options['SAVE_PREFIX'] + '_{epoch:02d}_{val_loss:.4f}.weights'
	weight_save = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=5, min_lr=options['MIN_LR'])
	print 'LOADING TRAINING DATA FROM: ', options['TRAIN']
	sys.stdout.flush()
	train_X, train_Y = generate_X_Y(options, 'TRAIN')
	train_gen = data_generator(train_X, train_Y, options)
	print 'LOADING VALIDATION DATA FROM: ', options['VAL']
	sys.stdout.flush()
	val_X, val_Y = generate_X_Y(options,'VAL', num_elem = 50000)		
	val_gen = data_generator(val_X, val_Y, options)
	print 'STARTING TRAINING ....'
	sys.stdout.flush()
	#steps_epoch = len(train_X) / options['BATCH_SIZE'] if len(train_X) % options['BATCH_SIZE'] == 0else (len(train_X) / options['BATCH_SIZE']) + 1
	steps_val = len(val_X) / options['BATCH_SIZE'] if len(val_X) % options['BATCH_SIZE'] == 0  else (len(val_X) / options['BATCH_SIZE']) + 1
	steps_epoch = 500
	#print steps_epoch
	#print steps_val
	model.fit_generator(train_gen, steps_per_epoch=steps_epoch, epochs=500, 
			  validation_data=val_gen, validation_steps=steps_val,
			  callbacks=[weight_save, reduce_lr])
	