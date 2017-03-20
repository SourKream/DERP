from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
import numpy as np

def tokenize(sent):
    return TreebankWordTokenizer().tokenize(sent)

def map_to_idx(x, vocab):
    return [vocab[w] if w in vocab else vocab['UNK'] for w in x]

def map_to_txt(x, vocab):
    textify = map_to_idx(x, inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {vocab[item]: item for item in vocab}

def load_data(filename, num_dat_points=-1):
    with open(filename,'rt') as f:    
        if num_dat_points==-1:
            dat = f.readlines()
        else:
            dat = []
            for i in range(num_dat_points):
                dat.append(f.readline())
                
    for i in range(len(dat)):
        dat[i] = [x.lower() for x in dat[i].strip().split('\t')] # !when making vocab all are lower case
    assert(all([len(x)==4 for x in dat]))
    dat_x = [x[:3] for x in dat]
    dat_y = [int(x[3]) for x in dat]
    
    return dat_x, dat_y

def data_generator(data_x, data_y, vocab_dict, count_vect, tfidf_transformer):
    assert(0 not in inverse_map(vocab_dict))  # 0 is for masking
    
    while True:
        for i in range(0, len(data_x), BATCH_SIZE):
            cur_batch_x = data_x[i:i+BATCH_SIZE]
            cur_batch_y = data_y[i:i+BATCH_SIZE]
            
            cur_batch_ctxt, cur_batch_gold_resp, cur_batch_alt_resp = zip(*cur_batch_x)
            for i in range(len(cur_batch_x)):
                cur_batch_gold_resp[i] = map_to_idx(tokenize(cur_batch_gold_resp[i]), vocab_dict)
                cur_batch_alt_resp[i] = map_to_idx(tokenize(cur_batch_alt_resp[i]), vocab_dict)
            
            # tfidf for context
            cur_batch_ctxt = count_vect.transform(cur_batch_ctxt)
            cur_batch_ctxt = tfidf_transformer.transform(cur_batch_ctxt)
            
            # indices for responses, 0 if nothing
            cur_batch_gold_resp_vec = pad_sequences(cur_batch_gold_resp, maxlen=MAX_RESP_LEN, value=vocab_dict['UNK'], padding='post', truncating='post')
            cur_batch_alt_resp_vec = pad_sequences(cur_batch_alt_resp, maxlen=MAX_RESP_LEN, value=vocab_dict['UNK'], padding='post', truncating='post')
            
            yield [cur_batch_ctxt.todense(), cur_batch_gold_resp_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)

