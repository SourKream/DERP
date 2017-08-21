from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from sklearn.externals import joblib
import cPickle as cp
from collections import Counter
import numpy as np
import os
import sys
sys.path.append('/home/bass/DERP/Code')
import tokenizer as tk

def my_tokenize(sent):
    return tk.my_tokenize(sent)

def map_to_ix(word, vocab, lower=True):
    return tk.map_to_ix(word,vocab,lower)


class WeightSave(Callback):
    def __init__(self, options):
        self.options = options
    def on_train_begin(self, logs={}):
        if self.options['LOAD_WEIGHTS']:
            assert 'MODEL_FILE' in self.options
            print('LOADING WEIGHTS FROM : ' + self.options['MODEL_FILE'])
            weights = joblib.load( self.options['MODEL_FILE'] )
            self.model.set_weights(weights)
    def on_epoch_end(self, epochs, logs = {}):
        cur_weights = self.model.get_weights()
        joblib.dump(cur_weights, self.options['SAVE_PREFIX'] + '_on_epoch_' + str(epochs) + '.weights')





def get_best_model_file(file_prefix, mode = "min"):
    '''
        this file acts as a helper function, called by get_best_model for most models
    '''
    def ends_with_correct_suffix(file, allowed_suffixes):
        for suffix in allowed_suffixes:
            if file.endswith(suffix):
                return True,suffix
        return False,None
    
    allowed_suffixes = set(['.hdf5','.weights'])
    mode = mode.lower()
    file_prefix = file_prefix.split('/')
    assert mode in set(["min", "max"])
    directory = '/'.join(file_prefix[:-1])
    model_prefix = file_prefix[-1]
    assert os.path.isdir(directory)
    best_model_metric = None
    best_model_file = None
    comparison_function = min if mode == "min" else max
    for file in os.listdir(directory):
        is_correct_suffix, model_suffix = ends_with_correct_suffix(file, allowed_suffixes)
        if file.startswith(model_prefix) and is_correct_suffix:
            assert model_suffix is not None
            # metric = float('.'.join(file.split('_')[-1].split('.')[:-1]))
            metric = float(file.rstrip(model_suffix).split('_')[-1])
            if best_model_metric is None:
                best_model_metric = metric
                best_model_file = file
            elif metric == comparison_function(best_model_metric, metric):
                best_model_metric = metric
                best_model_file = file
    assert best_model_file is not None
    print 'LOADING WEIGHTS FROM ...'
    print best_model_file
    sys.stdout.flush()
    return directory + '/' + best_model_file
    


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            # self.model.save_weights(filepath, overwrite=True)
                            cur_weights = self.model.get_weights()
                            cp.dump(cur_weights, open(filepath, 'wb'))
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    cur_weights = self.model.get_weights()
                    cp.dump(cur_weights, open(filepath,'wb'))
                else:
                    self.model.save(filepath, overwrite=True)


