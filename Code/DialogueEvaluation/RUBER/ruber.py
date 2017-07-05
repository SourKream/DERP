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
import cPickle as cp
from keras.engine.topology import Layer
from sklearn.externals import joblib
import numpy as np
import pdb
import theano.tensor as T
import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')
from base_utils import ModelCheckpoint
import argparse

def get_arguments():
	parser = argparse.ArgumentParser(description='Ruber model')
	parser.add_argument('-margin', action="store", default=0.5, dest="margin", type=float)
	opts = parser.parse_args(sys.argv[1:])	
	return opts

def max_margin_loss(y_true,y_pred):
	return K.mean( K.maximum(0., y_true - y_pred) ) # MARGIN - (y_pos - y_neg)
	#return K.mean( K.maximum(0., y_true[0] - y_pred[0] + y_pred[1]) )

class BilinearLayer(Layer):
	def __init__(self, dense_dim, **kwargs):
		self.dense_dim = dense_dim
		super(BilinearLayer,self).__init__(**kwargs)
	
	def build(self, input_shapes):
		self.ctx_dim = input_shapes[0][1]
		self.resp_dim = input_shapes[1][1]
		self.ctx_dense = self.add_weight(name = 'ctx_dense', shape=(self.ctx_dim, self.resp_dim), initializer='glorot_uniform', trainable=True)
	
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
	ctxt_gru = Bidirectional(GRU(options['CTXT_GRU_HIDDEN_STATE']))
	encoded_ctxt = ctxt_gru(ctxt_embed)
	resp_gru = Bidirectional(GRU(options['RESP_GRU_HIDDEN_STATE']))
	encoded_resp = resp_gru(resp_embed)
	bilinear_module = BilinearLayer(options['CTXT_GRU_HIDDEN_STATE']) 
	bilinear_ctxt_resp = bilinear_module([encoded_ctxt, encoded_resp])
	merged_vec = keras.layers.concatenate([encoded_ctxt, encoded_resp, bilinear_ctxt_resp], axis=-1)
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
	if options['USE_GLOVE']:
		options['EMBEDDING_MATRIX'] = np.load(open(options['EMBED_PATH'],'r'))
	print 'MARGIN VALUE: ', options['MARGIN']
	sys.stdout.flush()
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
	val_X, val_Y = generate_X_Y(options,'VAL', num_elem=50000)		
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
		