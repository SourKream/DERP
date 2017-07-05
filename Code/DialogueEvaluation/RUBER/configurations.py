from sklearn.externals import joblib
import sys
sys.path.append('/home/bass/DERP/Code/DialogueEvaluation')
import base_configurations as bc

def get_options(args):
	options = bc.get_options()
	'''
		base configurations contains the following :				
		'MIN_LR'
		'BATCH_SIZE'
		'MAX_CTX_LEN'
		'MAX_RESP_LEN'
		'DROPOUT'
		'CTXT_GRU_HIDDEN_STATE'
		'RESP_GRU_HIDDEN_STATE'
		'DENSE_HIDDEN_STATE'	
		'DATA_DIR'
		'VOCAB_PATH'
		'VOCAB'
		'EMBEDDING_DIM'
		'EMBED_PATH'
		'USE_GLOVE'
	'''
	if hasattr(args, 'resume_training'):
		options['LOAD_WEIGHTS'] = args.resume_training # True then resumes training from a pre-exting model
	else:
		options['LOAD_WEIGHTS'] = False
	options['BILINEAR'] = False	
	# Now the vocabulary 
	options['MARGIN'] = args.margin
	# Now the data files
	options['TRAIN'] = options['DATA_DIR'] + 'RUBER_train.txt'
	options['VAL'] =  options['DATA_DIR'] + 'RUBER_val.txt'

	# Now the model save path
	options['SAVE_DIR'] = "/scratch/cse/btech/cs1130773/DERPData/Models_Clean/RUBER/RUBER/" 
	options['SAVE_PREFIX'] = options['SAVE_DIR'] + "model"
	if options['LOWER']:
		options['SAVE_PREFIX'] += '_LOWER_TRUE'
	options['SAVE_PREFIX'] += '_MAX_CTX_LEN_' + str(options['MAX_CTX_LEN'])
	options['SAVE_PREFIX'] += '_MAX_RESP_LEN_' + str(options['MAX_RESP_LEN'])
	options['SAVE_PREFIX'] += '_MAX_RESP_LEN_' + str(options['MAX_RESP_LEN'])
	options['SAVE_PREFIX'] += '_EMBEDDING_DIM_' + str(options['EMBEDDING_DIM'])
	options['SAVE_PREFIX'] += '_CTX_GRU_DIM_' + str(options['CTXT_GRU_HIDDEN_STATE'])	
	options['SAVE_PREFIX'] += '_RESP_GRU_DIM_' + str(options['RESP_GRU_HIDDEN_STATE'])
	options['SAVE_PREFIX'] += '_DENSE_DIM_' + str(options['DENSE_HIDDEN_STATE'])
	options['SAVE_PREFIX'] += '_MARGIN_' + str(options['MARGIN'])
	

	return options
