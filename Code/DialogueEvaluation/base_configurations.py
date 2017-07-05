from sklearn.externals import joblib
def get_options():
	options = {}		
	options['MIN_LR'] = 0.0001
	# Network configuration
	options['BATCH_SIZE'] = 256
	options['MAX_CTX_LEN'] = 500
	options['MAX_RESP_LEN'] = 50	
	options['DROPOUT'] = 0.0
	
	# Hidden state details
	options['CTXT_GRU_HIDDEN_STATE'] = 512
	options['RESP_GRU_HIDDEN_STATE'] = 300
	options['DENSE_HIDDEN_STATE'] = 100
	
	# Now the vocabulary 
	options['DATA_DIR'] = '/home/bass/DERPData/'
	options['VOCAB_PATH'] = options['DATA_DIR'] + 'vocab_50k'
	options['VOCAB'] = joblib.load(options['VOCAB_PATH'])

	# Embedding layer details
	options['EMBEDDING_DIM'] = 300
	options['EMBED_PATH'] = options['DATA_DIR'] + 'glove_embedding_vocab50k.npy'
	options['USE_GLOVE'] = True
	options['LOWER'] = False
	
	# All models are tested against same test files
	# 1.1 Reddit internal
	options['REDDIT_INTERNAL_TEST_FILE'] = '/scratch/cse/btech/cs1130773/DERPData/Data/test.txt'
	
	options['UBUNTU_HRED'] = '/scratch/cse/btech/cs1130773/DERPData/Ubuntu/HRED_Ubuntu_responses.txt'
	options['UBUNTU_HRED_ActEnt'] = '/scratch/cse/btech/cs1130773/DERPData/Ubuntu/HRED_ActEnt_Ubuntu_responses.txt'
	options['UBUNTU_MRRNN_ActEnt'] = '/scratch/cse/btech/cs1130773/DERPData/Ubuntu/MRRNN_ActEnt_Ubuntu_responses.txt'
	options['UBUNTU_MRRNN_Noun'] = '/scratch/cse/btech/cs1130773/DERPData/Ubuntu/MRRNN_Noun_Ubuntu_responses.txt'
	return options
