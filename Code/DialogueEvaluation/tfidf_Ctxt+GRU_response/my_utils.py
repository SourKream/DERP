from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from tfidfctxt_gruresp import *
from config import *
import numpy as np
from scipy.sparse import csr_matrix
import re
import pdb

tokenizer_exceptions = [u'(amazon\\.com)', u'(google\\.com)', u'(a\\.k\\.a\\.)', u'(r\\.i\\.p\\.)', u'(states\\.)', u'(a\\.k\\.a)', u'(r\\.i\\.p)', u'(corps\\.)', u'(ph\\.d\\.)', u'(corp\\.)', u'(j\\.r\\.)', u'(b\\.s\\.)', u'(alex\\.)', u'(d\\.c\\.)', u'(b\\.c\\.)', u'(bros\\.)', u'(j\\.j\\.)', u'(mins\\.)', u'(\\.\\.\\.)', u'(dept\\.)', u'(a\\.i\\.)', u'(u\\.k\\.)', u'(c\\.k\\.)', u'(p\\.m\\.)', u'(reps\\.)', u'(prof\\.)', u'(p\\.s\\.)', u'(l\\.a\\.)', u'(i\\.e\\.)', u'(govt\\.)', u'(u\\.s\\.)', u'(t\\.v\\.)', u'(a\\.m\\.)', u'(cons\\.)', u'(e\\.g\\.)', u'(j\\.k\\.)', u'(ave\\.)', u'(gen\\.)', u'(feb\\.)', u'(mrs\\.)', u'(etc\\.)', u'(vol\\.)', u'(gov\\.)', u'(sec\\.)', u'(nov\\.)', u'(hrs\\.)', u'(sgt\\.)', u'(mon\\.)', u'(jan\\.)', u'(min\\.)', u'(pts\\.)', u'(rev\\.)', u'(inc\\.)', u'(est\\.)', u'(cal\\.)', u'(sat\\.)', u'(dec\\.)', u'(rep\\.)', u'(lbs\\.)', u'(mr\\.)', u'(jr\\.)', u'(km\\.)', u'(dc\\.)', u'(p\\.s)', u'(pp\\.)', u'(ex\\.)', u'(op\\.)', u'(co\\.)', u'(sp\\.)', u'(u\\.s)', u'(vs\\.)', u'(kg\\.)', u'(ms\\.)', u'(iv\\.)', u'(ca\\.)', u'(sr\\.)', u'(oz\\.)', u'(bc\\.)', u'(dr\\.)', u'(ga\\.)', u'(lb\\.)', u'(mi\\.)', u'(ad\\.)', u'(ft\\.)', u'(e\\.g)', u'(ed\\.)', u'(sc\\.)', u'(lt\\.)', u'(va\\.)', u'(la\\.)', u'(mt\\.)', u'(i\\.e)', u'(st\\.)', u'(mo\\.)']

def my_tokenize(sent):
    return [''.join(x) for x in re.findall("|".join(tokenizer_exceptions)+"|([0-9]+)|('\w{1,2}[^\w])|([\w]+)|([.,!?;'])",sent)]

def load_sparse_csr(filename):
    # http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

class WeightSave(Callback):
    # for checkpointing models, since there's some issue with the keras code
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
                                 
def map_to_idx(x, vocab):
    return [vocab[w] if w in vocab else vocab['UNK'] for w in x]

def map_to_txt(x, vocab):
    textify = map_to_idx(x, inverse_map(vocab))
    return ' '.join(textify)
    
def inverse_map(vocab):
    return {vocab[item]: item for item in vocab}

def load_data_raw(filename, num_dat_points=-1):
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

def data_generator_raw(data_x, data_y, vocab_dict, count_vect, tfidf_transformer):
    assert(0 not in inverse_map(vocab_dict))  # 0 is for masking
    
    while True:
        for i in range(0, len(data_x), BATCH_SIZE):
            cur_batch_x = data_x[i:i+BATCH_SIZE]
            cur_batch_y = data_y[i:i+BATCH_SIZE]
            
            cur_batch_ctxt, cur_batch_gold_resp, cur_batch_alt_resp = [list(x) for x in zip(*cur_batch_x)]
            
            for j in range(len(cur_batch_x)):
                cur_batch_gold_resp[j] = map_to_idx(my_tokenize(cur_batch_gold_resp[j]), vocab_dict)
                cur_batch_alt_resp[j] = map_to_idx(my_tokenize(cur_batch_alt_resp[j]), vocab_dict)
            
            # tfidf for context
            cur_batch_ctxt = count_vect.transform(cur_batch_ctxt)
            cur_batch_ctxt = tfidf_transformer.transform(cur_batch_ctxt)
            
            # indices for responses, 0 if nothing
            cur_batch_gold_resp_vec = pad_sequences(cur_batch_gold_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            cur_batch_alt_resp_vec = pad_sequences(cur_batch_alt_resp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            
            yield [cur_batch_ctxt.todense(), cur_batch_gold_resp_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)

def data_generator_preprocessed(data_ctxt_tfidfed, data_gold_resp_preprocessed, data_alt_resp_preprocessed, data_y):
    """
    data_ctxt_tfidfed : N x len(vocab) sparse vector of count vectorized + tfidf_transformed context 
    data_gold_resp_preprocessed : array of length N where each item is a list of indexes obtained by tokenizing+indexing gold responses
    data_alt_resp_preprocessed : array of length N where each item is a list of indexes obtained by tokenizing+indexing alt responses
    
    padding is required for responses.
    """
    
    while True:
        for i in range(0, len(data_y), BATCH_SIZE):
            cur_batch_ctxt_tfidfed = data_ctxt_tfidfed[i:i+BATCH_SIZE]
            cur_batch_gold_resp_pp = data_gold_resp_preprocessed[i:i+BATCH_SIZE]
            cur_batch_alt_resp_pp = data_alt_resp_preprocessed[i:i+BATCH_SIZE]
            cur_batch_y = data_y[i:i+BATCH_SIZE]
            
            cur_batch_gold_resp_vec = pad_sequences(cur_batch_gold_resp_pp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            cur_batch_alt_resp_vec = pad_sequences(cur_batch_alt_resp_pp, maxlen=MAX_RESP_LEN, value=0, padding='post', truncating='post')
            
            yield [cur_batch_ctxt_tfidfed.todense(), cur_batch_gold_resp_vec, cur_batch_alt_resp_vec], np.array(cur_batch_y)
    
            
def test_routine(model):
    assert load_model_path != ''

    print 'Loading Data ...'
    test_x, test_y = load_data_raw(test_file, -1)
    test_gen = data_generator_raw(test_x, test_y, vocab_dict)
    print 'Restoring Weights ...'
    model.set_weights(joblib.load(load_model_path))

    print 'Testing ...'
    probs = model.predict_generator(test_gen, steps = len(test_x)/BATCH_SIZE).flatten()
    y = np.array(test_y)[:len(probs)]
    pred = np.floor(probs + 0.5)
    f = open(save_pred_path + 'preds_' + model_name + '.pkl', 'w')
    cPickle.dump(y, f)
    cPickle.dump(probs, f)
    cPickle.dump(pred, f)

    from sklearn.metrics import *
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    accuracy = accuracy_score(y, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y, pred, average = 'binary')
    confusion_matrix = confusion_matrix(y, pred)
    print 'Accuracy: ' + '%.2f' % accuracy
    print 'Precision: ' + '%.2f' % precision
    print 'Recall: ' + '%.2f' % recall
    print 'F-Score: ' + '%.2f' % f_score
    print confusion_matrix
    precs, recs, thresholds = precision_recall_curve(y, probs)

    plt.clf()
    plt.plot(recs, precs, lw=2, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc = 'lower left')
    plt.savefig(save_pred_path + 'pr_curve_' + model_name + '.png')
    