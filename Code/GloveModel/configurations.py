# model params
FILENAME = './tiny_glove.txt'
MAX_WORD_LEN = 10
CHAR_EMBED_SIZE = 50
GRU_SIZE = 300
OUTPUT_SIZE = 300
BATCH_SIZE = 256

save_model_path = '/scratch/cse/dual/cs5130275/DERP/GloveModel/Models/' + 'Output_' + str(OUTPUT_SIZE) + '_GRU_' + str(GRU_SIZE) + '_MAXLEN_' + str(MAX_WORD_LEN) + '_CHAREMBED_' + str(CHAR_EMBED_SIZE) + '_FILE_' + ''.join(str(FILENAME).split('/')) 
