from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from sklearn.externals import joblib
from collections import Counter
from configurations import *
import numpy as np
import re
import pdb


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

tokenizer_exceptions = [u'(amazon\\.com)', u'(google\\.com)', u'(a\\.k\\.a\\.)', u'(r\\.i\\.p\\.)', u'(states\\.)', u'(a\\.k\\.a)', u'(r\\.i\\.p)', u'(corps\\.)', u'(ph\\.d\\.)', u'(corp\\.)', u'(j\\.r\\.)', u'(b\\.s\\.)', u'(alex\\.)', u'(d\\.c\\.)', u'(b\\.c\\.)', u'(bros\\.)', u'(j\\.j\\.)', u'(mins\\.)', u'(\\.\\.\\.)', u'(dept\\.)', u'(a\\.i\\.)', u'(u\\.k\\.)', u'(c\\.k\\.)', u'(p\\.m\\.)', u'(reps\\.)', u'(prof\\.)', u'(p\\.s\\.)', u'(l\\.a\\.)', u'(i\\.e\\.)', u'(govt\\.)', u'(u\\.s\\.)', u'(t\\.v\\.)', u'(a\\.m\\.)', u'(cons\\.)', u'(e\\.g\\.)', u'(j\\.k\\.)', u'(ave\\.)', u'(gen\\.)', u'(feb\\.)', u'(mrs\\.)', u'(etc\\.)', u'(vol\\.)', u'(gov\\.)', u'(sec\\.)', u'(nov\\.)', u'(hrs\\.)', u'(sgt\\.)', u'(mon\\.)', u'(jan\\.)', u'(min\\.)', u'(pts\\.)', u'(rev\\.)', u'(inc\\.)', u'(est\\.)', u'(cal\\.)', u'(sat\\.)', u'(dec\\.)', u'(rep\\.)', u'(lbs\\.)', u'(mr\\.)', u'(jr\\.)', u'(km\\.)', u'(dc\\.)', u'(p\\.s)', u'(pp\\.)', u'(ex\\.)', u'(op\\.)', u'(co\\.)', u'(sp\\.)', u'(u\\.s)', u'(vs\\.)', u'(kg\\.)', u'(ms\\.)', u'(iv\\.)', u'(ca\\.)', u'(sr\\.)', u'(oz\\.)', u'(bc\\.)', u'(dr\\.)', u'(ga\\.)', u'(lb\\.)', u'(mi\\.)', u'(ad\\.)', u'(ft\\.)', u'(e\\.g)', u'(ed\\.)', u'(sc\\.)', u'(lt\\.)', u'(va\\.)', u'(la\\.)', u'(mt\\.)', u'(i\\.e)', u'(st\\.)', u'(mo\\.)']

def my_tokenize(sent):
    return [''.join(x) for x in re.findall("|".join(tokenizer_exceptions)+"|([0-9]+)|('\w{1,2}[^\w])|([\w]+)|([.,!?;'])",sent)]

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

def data_generator(data_x, data_y, vocab_dict):
    assert(0 not in inverse_map(vocab_dict))  # 0 is for masking 
    while True:
        for i in range(0, len(data_x), BATCH_SIZE):
            cur_batch_x = data_x[i:i+BATCH_SIZE]
            cur_batch_y = data_y[i:i+BATCH_SIZE]
 
            cur_batch_ctxt, cur_batch_gold_resp, cur_batch_alt_resp = [list(x) for x in zip(*cur_batch_x)]
            for i in range(len(cur_batch_x)):
                cur_batch_alt_resp[i] = map_to_idx(my_tokenize(cur_batch_alt_resp[i]), vocab_dict)
                cur_batch_ctxt[i] = map_to_idx(my_tokenize(cur_batch_ctxt[i]), vocab_dict)
            # indices for responses, 0 if nothing
            cur_batch_alt_resp_vec = pad_sequences(cur_batch_alt_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            cur_batch_ctxt_vec = pad_sequences(cur_batch_ctxt, maxlen=MAX_CTX_LEN, value=0, padding='post', truncating='post')
            yield [cur_batch_ctxt_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)
