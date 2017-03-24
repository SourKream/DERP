import keras
from keras.layers import Input, GRU, Embedding, Dense, Activation, add, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import *
import theano
import theano.tensor as T

from configurations import *
from utils import *

from sklearn.externals import joblib
import numpy as np
import pdb

import sys

class SingleAttentionLayer(Layer):
    def __init__(self, dense_dim, return_att = False, **kwargs):
        self.return_att = return_att
        self.dense_dim = dense_dim
        super(SingleAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ctxt_dim = input_shape[0][2]
        self.resp_dim = input_shape[1][1]
        self.ctxt_dense = self.add_weight(shape=(self.ctxt_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.resp_dense = self.add_weight(shape=(self.resp_dim, self.dense_dim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.alpha_dense = self.add_weight(shape=(self.dense_dim, 1), initializer='glorot_uniform', trainable=True) # n_dim x 1
        super(SingleAttentionLayer, self).build(input_shape)

    def call(self, input_tensors, mask = None):
        ''' wbw attention layer:
        :param ctxt (input_tensors[0]) : batch_size x T x ctxt_dim
        :param resp (input_tensors[1]) : batch_size x resp_dim
        '''

        ctxt = input_tensors[0]
        resp = input_tensors[1]
        ctxt_mask = mask[0]

        resp_w = T.dot(resp, self.resp_dense) # bt_sz x dense_dim
        ctxt_w = T.dot(ctxt, self.ctxt_dense) # bt_sz x T x dense_dim
        resp_w_rep = resp_w[:,None,:] # bt_sz x T x dense_dim
        pre_alpha = T.tanh(ctxt_w + resp_w_rep) # bt_sz x T x dense_dim
        unnorm_alpha = T.dot(pre_alpha, self.alpha_dense).flatten(2) # bt_sz x T
        if ctxt_mask:
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

# def single_attention(ctxt, resp, ctxt_dense, resp_dense, alpha_dense):
#     pre_alpha = Activation('tanh')(add([RepeatVector(MAX_CTX_LEN)(resp_dense(resp)), TimeDistributed(ctxt_dense)(ctxt)]))
#     alpha = Activation("softmax")(Flatten()(TimeDistributed(alpha_dense)(pre_alpha)))
#     alpha = Reshape((MAX_CTX_LEN, 1), input_shape=(MAX_CTX_LEN,))(alpha)
#     permuted_ctxt = K.permute_dimensions(ctxt, (0,2,1))
#     return K.T.batched_dot(permuted_ctxt, alpha)

def create_model():
    ctxt = Input(shape=(MAX_CTX_LEN,))
    gold_resp = Input(shape=(MAX_RESP_LEN,))
    alt_resp = Input(shape=(MAX_RESP_LEN,))
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE+2, mask_zero=True)   # +1 for 'UNK', +1 for mask (0 can't be used)
    
    ctxt_emb = embedding(ctxt)


    gold_resp_emb = embedding(gold_resp)
    alt_resp_emb = embedding(alt_resp)

    ctxt_gru = Bidirectional(GRU(CTXT_GRU_HIDDEN_STATE, return_sequences = True))
    encoded_ctxt = ctxt_gru(ctxt_emb)

    shared_gru = Bidirectional(GRU(RESP_GRU_HIDDEN_STATE))
    encoded_gold_resp = shared_gru(gold_resp_emb)
    encoded_alt_resp = shared_gru(alt_resp_emb)

    # shared_ctxt_dense = Dense(CTXT_GRU_HIDDEN_STATE)
    # shared_resp_dense = Dense(CTXT_GRU_HIDDEN_STATE)
    # shared_alpha_dense = Dense(1)
    # attended_gold_ctxt = single_attention(encoded_ctxt, encoded_gold_resp, shared_ctxt_dense, shared_resp_dense, shared_alpha_dense)
    # attended_alt_ctxt = single_attention(encoded_ctxt, encoded_alt_resp, shared_ctxt_dense, shared_resp_dense, shared_alpha_dense)

    attention_module = SingleAttentionLayer(CTXT_GRU_HIDDEN_STATE, return_att = True)
    attended_gold_ctxt, gold_alpha = attention_module([encoded_ctxt, encoded_gold_resp])
    attended_alt_ctxt, ctxt_alpha = attention_module([encoded_ctxt, encoded_alt_resp])

    merged_vector = keras.layers.concatenate([attended_gold_ctxt, attended_alt_ctxt, encoded_gold_resp, encoded_alt_resp], axis=-1)
    if DROPOUT > 0.0:
        merged_vector = Dropout(DROPOUT)(merged_vector)
    merged_vector = Dense(DENSE_HIDDEN_STATE, activation='tanh')(merged_vector)
    if DROPOUT > 0.0:
        merged_vector = Dropout(DROPOUT)(merged_vector)    
    predictions = Dense(1, activation='sigmoid')(merged_vector)
    
    model = Model(inputs=[ctxt, gold_resp, alt_resp], outputs=predictions)
    adam = Adam(clipnorm=1.)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model

if __name__=='__main__':

    count_vect_vocab = joblib.load(count_vect_vocab_file)
    # tfidf_transformer = joblib.load(tfidf_transformer_file)
    
    assert(len(count_vect_vocab)==VOCAB_SIZE)
    # count_vect = CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)
    # count_vect.vocabulary_ = count_vect_vocab
    vocab_dict = {x:i+1 for i,x in enumerate(count_vect_vocab)}     # +1 since 0 is for masking
    vocab_dict['UNK'] = len(vocab_dict)+1
    inv_vocab = {vocab_dict[x]:x for x in vocab_dict}

    model = create_model()
    weight_save = WeightSave()
    weight_save.model_file = save_model_path
    weight_save.load_model_path = load_model_path

    if len(sys.argv) > 1 and sys.argv[1] == 'Test':
        assert load_model_path != ''

        print 'Loading Data ...'
        test_x, test_y = load_data(test_file, -1)
        test_gen = data_generator(test_x, test_y, vocab_dict)
        print 'Restoring Weights ...'
        model.set_weights(joblib.load(load_model_path))

        print 'Testing ...'
        probs = model.predict_generator(test_gen, steps = len(test_x)/BATCH_SIZE).flatten()
        y = np.array(test_y)[:len(probs)]
        pred = np.floor(probs + 0.5)
        f = open(save_pred_path + 'preds.pkl', 'w')
        cPickle.dump(y, f)
        cPickle.dump(probs, f)
        cPickle.dump(pred, f)

        from sklearn.metrics import *
        import matplotlib.pyplot as plt

        accuracy = accuracy_score(y, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(y, pred, average = 'binary')
        confusion_matrix = confusion_matrix(y, pred)
        print 'Accuracy: ' + str(accuracy)
        print 'Precision: ' + str(precision)
        print 'Recall: ' + str(recall)
        print 'F-Score: ' + str(f_score)
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
        plt.savefig(save_pred_path + 'pr_curve.png')

    else:
        train_x, train_y = load_data(train_file, TRAIN_SIZE)
        val_x, val_y = load_data(val_file, 10000)

        train_gen = data_generator(train_x, train_y, vocab_dict)
        val_gen = data_generator(val_x, val_y, vocab_dict)

        # model.fit_generator(train_gen, steps_per_epoch=len(train_x)/BATCH_SIZE, epochs=10)
    
        model.fit_generator(train_gen, steps_per_epoch=500, epochs=500, validation_data=val_gen, validation_steps=len(val_x)/BATCH_SIZE, callbacks=[weight_save])
