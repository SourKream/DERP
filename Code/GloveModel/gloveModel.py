from keras.layers import *
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *
from keras.optimizers import *
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from theano import tensor as T
from sklearn.externals import joblib
from configurations import *
import numpy as np
import keras

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

def map_to_idx(x, vocab):
    return [vocab[w] if w in vocab else vocab['UNK'] for w in x]

def create_model():
    word = Input(shape=(MAX_WORD_LEN,))
    word_emb = Embedding(output_dim=CHAR_EMBED_SIZE, input_dim=LEN_CHAR_VOCAB)(word)
    word_glove = Bidirectional(GRU(GRU_SIZE, return_sequences = False))(word_emb)
    word_glove = Dense(OUTPUT_SIZE, activation='tanh')(word_glove)
    
    model = Model(inputs=word, outputs=word_glove)
    model.compile(optimizer=Adam(clipnorm=1.), loss='cosine_proximity')
    model.summary()
    return model

if __name__=='__main__':

    ## Load Data
    X = []
    y = []
    for line in open(FILENAME):
        line = line.strip().split(' ')
        word = line[0]
        word_vector = [float(x) for x in line[1:]]
        X.append(word)
        y.append(word_vector)

    characters = set([])
    for word in X:
        characters.update(list(word))

    char_vocab = {x:(y+1) for y,x in enumerate(sorted(list(characters)))}
    char_vocab['<PAD>'] = 0
    char_vocab['<EOW>'] = len(char_vocab)
    char_vocab['UNK'] = len(char_vocab)

    LEN_CHAR_VOCAB = len(char_vocab)
    print "Number of characters : ", LEN_CHAR_VOCAB

    x = []
    for word in X:
        word = list(word) + ['<EOW>']
        x.append(map_to_idx(word, char_vocab))

    x = pad_sequences(x, maxlen=MAX_WORD_LEN, value=char_vocab['<PAD>'], padding='post', truncating='post')
    y = np.matrix(y)

    model = create_model()
    weight_save = WeightSave()
    weight_save.model_file = './' + filename
    weight_save.load_model_path = False
    
    num_samples = x.shape[0]
    num_train_samples = int(0.995*num_samples)
    print "Training Samples = ", num_train_samples
    print "Val Samples = ", len(x) - num_train_samples
    train_x, train_y = x[:num_train_samples,:], y[:num_train_samples,:]
    val_x, val_y = x[num_train_samples:,:], y[num_train_samples:,:]

    model.fit(x = train_x, y = train_y, batch_size = BATCH_SIZE, epochs = 10, validation_data = (val_x, val_y), callbacks = [weight_save])

