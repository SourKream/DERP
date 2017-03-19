#TODO: Take care of imports
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import *
from keras.layers.core import *
import argparse
import sys
import numpy
sys.path.append('/home/cse/dual/cs5130275/DERP/Code')

import SkipThoughts.skipthoughts as skt
def get_params():
	parser = argparse.ArgumentParser(description='Logistic regression')    
	parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
	parser.add_argument('-batch', action="store", default=256, dest="batch_size", type=int)		
	parser.add_argument('-lr', action="store", default=0.01, dest="lr", type=float)
	parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
	parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
	parser.add_argument('-l2', action="store", default=0.0003, dest="l2", type=float)	
	parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
	parser.add_argument('-n_dim', action="store", default=14400, dest="n_dim", type=int)
	opts = parser.parse_args(sys.argv[1:])	
	print "epochs", opts.epochs
	print "batch_size", opts.batch_size	
	print "regularization factor", opts.l2	
	return opts

def build_model(options):
	inputs = Input(shape=(options.n_dim,), name='inputs')
	output = Dense(1, activation='sigmoid', name='output')(inputs)
	model = Model(input = inputs, output = output)
	model.summary()
	model.compile(loss='binary_crossentropy', 
				optimizer=Adam(options.lr),
				metrics=['accuracy'])
	return model

class WeightSave(Callback):
	def setModelFile(self, model_file):
		self.model_file = model_file
	def on_epoch_end(self, epochs, logs={}):
		self.model.save_weights(model_file + "_on_epoch_" + str(epochs) + ".weights")
# Local Stuff 
#def encode(model, data_list, verbose = False):
#	return numpy.random.normal(size = (len(data_list), 4800))
# End Local Stuff
		
def load_data(data_file):
	data = {'context':[],'gold':[], 'predicted':[],'label':[]}
	with open(data_file) as f:
		for line in f:
			line = line.strip().split('\t')
			data['context'].append(line[0])
			data['gold'].append(line[1])
			data['predicted'].append(line[2])
			data['label'].append(eval(line[3]))
	#data['Skip_model'] = None
	data['Skip_model'] = skt.load_model()
	return data

def data_generator(data, options):
	batch_size = options.batch_size
	while 1:
		for idx in xrange(0, batch_size, len(data['context'])):
			start_idx = idx
			end_idx = idx + batch_size if idx + batch_size < len(data['context']) else len(data['context'])
			#context_t = encode(data['Skip_model'], data['context'][start_idx:end_idx] , verbose = False )
			context_t = skt.encode(data['Skip_model'], data['context'][start_idx:end_idx] , verbose = False, batch_size = batch_size )
			#gold_t = encode(data['Skip_model'], data['gold'][start_idx:end_idx] , verbose = False )
			gold_t = skt.encode(data['Skip_model'], data['gold'][start_idx:end_idx] , verbose = False, batch_size = batch_size )
			#predicted_t = encode(data['Skip_model'], data['predicted'][start_idx:end_idx] , verbose = False )
			predicted_t = skt.encode(data['Skip_model'], data['predicted'][start_idx:end_idx] , verbose = False, batch_size = batch_size )
			label_t = numpy.array(data['label'][start_idx:end_idx])
			data_t = numpy.concatenate((context_t, predicted_t, gold_t), axis = 1)
			yield ({'inputs':data_t}, {'output':label_t} )
			

if __name__=="__main__":
	options = get_params()
	data_dir = None
	model_dir = None
	if options.local :
		data_dir = "../../"
		model_dir = "../../"
	else:
		data_dir = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/DatasetWithPruning7M/LogRegData/"
		model_dir = "/home/cse/dual/cs5130275/scratch/DERP/Models/LogRegModels/" 
	model_file = model_dir + "LogRegModelFile_"
	data_file = data_dir + "train_100K_clean.txt"
	data = load_data(data_file)
	batch_size = options.batch_size	
	if options.load_save:
		epoch2load = 1
		model = build_model(options)				
		model.load_weights(model_file + str(epoch2load) + ".model")		
	else:
		model = build_model(options)
		save_weights = WeightSave()
		save_weights.setModelFile(model_file)
		history = model.fit_generator( data_generator(data, options),
										 epochs = options.epochs,
										 steps_per_epoch = len(data['context']) / options.batch_size,
										 validation_data = None,
										 callbacks = [save_weights])



