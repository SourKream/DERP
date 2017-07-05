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
	# Now the data files
	if hasattr(args, 'bad_grammar') and args.bad_grammar:
		options['TRAIN'] = options['DATA_DIR'] + 'train_with_incorrect_grammar.txt'
		options['VAL'] =  options['DATA_DIR'] + 'val_with_incorrect_grammar.txt'
	else:
		#TODO : change this
		options['TRAIN'] = options['DATA_DIR'] + 'train.txt'
		options['VAL'] =  options['DATA_DIR'] + 'val.txt'
		# options['TEST'] = options['DATA_DIR'] + 'test.txt'

	# Now the model save path	
	options['SAVE_DIR'] = "/scratch/cse/btech/cs1130773/DERPData/Models_Clean/DERP/GRU_CTXT_GRU_RESP_ATTN/" 
	options['SAVE_PREFIX'] = options['SAVE_DIR'] 
	if hasattr(args, 'bad_grammar') and args.bad_grammar:
		options['SAVE_PREFIX'] += 'model_with_bad_grammar'
	else:
		options['SAVE_PREFIX'] += 'model'
	if options['LOWER']:
		options['SAVE_PREFIX'] += '_LOWER_TRUE'
	options['SAVE_PREFIX'] += '_CTXT_GRU_DIM_' + str(options['CTXT_GRU_HIDDEN_STATE'])
	options['SAVE_PREFIX'] += '_RESP_GRU_DIM_' + str(options['RESP_GRU_HIDDEN_STATE'])
	options['SAVE_PREFIX'] += '_HIDDEN_DIM_' + str(options['DENSE_HIDDEN_STATE'])
	options['SAVE_PREFIX'] += '_MAX_CTX_LEN_' + str(options['MAX_CTX_LEN'])
	options['SAVE_PREFIX'] += '_MAX_RESP_LEN_' + str(options['MAX_RESP_LEN'])
	options['SAVE_PREFIX'] += '_DROPOUT_' + str(options['DROPOUT']) + '_'


	return options
