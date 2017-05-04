import numpy as np

from keras.layers import GRU
from keras import backend as K
from keras.engine import InputSpec

import theano
from theano import tensor as T

class AttentionDecoderGRU(GRU):

    def __init__(self, units, attn_dense_dim, return_att = False, **kwargs):
        super(AttentionDecoderGRU, self).__init__(units, **kwargs)
        self.return_att = return_att
        self.attn_dense_dim = attn_dense_dim
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.state_spec = [InputSpec(shape=(None, self.units)), InputSpec(ndim=2)]


    def build(self, input_shape):
        self.ctxt_dim = input_shape[1][2]
        self.hidden_dense = self.add_weight(shape=(self.units, self.attn_dense_dim), 
                                            initializer='glorot_uniform',
                                            name = 'hidden_dense')
        self.ctxt_dense = self.add_weight(shape=(self.ctxt_dim, self.attn_dense_dim), 
                                          initializer='glorot_uniform', 
                                          name = 'ctxt_dense')
        self.alpha_dense = self.add_weight(shape=(self.attn_dense_dim, 1), 
                                           initializer='glorot_uniform', 
                                           name = 'alpha_dense')
        
        batch_size = input_shape[0][0] if self.stateful else None
        self.input_dim = input_shape[0][2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))
        self.input_spec[1] = InputSpec(shape=(batch_size, None, self.ctxt_dim))
        self.state_spec[1] = InputSpec(shape=(None, self.ctxt_dim))

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.context_kernel = self.add_weight(shape=(self.ctxt_dim, self.units * 3),
                                              name='context_kernel',
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.context_kernel_z = self.context_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.context_kernel_r = self.context_kernel[:, self.units: self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]
        self.context_kernel_h = self.context_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def preprocess_input(self, inputs, training = None):
        return super(AttentionDecoderGRU, self).preprocess_input(inputs, training)

    def get_constants(self, inputs, mask, training = None):
        constants = super(AttentionDecoderGRU, self).get_constants(inputs, training)
        constants.append(inputs[1])
        constants.append(mask[1])
        return constants

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
        initial_state = [initial_state for _ in range(len(self.states))]
        return initial_state

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.

        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs[0])
            initial_state[-1] = K.zeros_like(inputs[1])
            initial_state[-1] = K.sum(initial_state[-1], axis=(1, 2))  # (samples,)
            initial_state[-1] = K.expand_dims(initial_state[-1])  # (samples, 1)
            initial_state[-1] = K.tile(initial_state[-1], [1, self.ctxt_dim])  # (samples, ctxt_dim)

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs[0])
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, mask, training=None)
        preprocessed_input = self.preprocess_input(inputs[0], training=None)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return [outputs, states[1]]
        else:
            return last_output

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        attended_ctxt = states[1]
        dp_mask = states[2]  # dropout matrices for recurrent units
        rec_dp_mask = states[3]
        ctxt = states[4]
        ctxt_mask = states[5]

        # hidden_w = T.dot(h_tm1, self.hidden_dense) # bt_sz x attn_dense_dim
        # ctxt_w = T.dot(ctxt, self.ctxt_dense) # bt_sz x T x attn_dense_dim
        # hidden_w_rep = hidden_w[:,None,:] # bt_sz x T x attn_dense_dim
        # pre_alpha = T.tanh(ctxt_w + hidden_w_rep) # bt_sz x T x attn_dense_dim
        # unnorm_alpha = T.dot(pre_alpha, self.alpha_dense).flatten(2) # bt_sz x T
        # if ctxt_mask:
        #     unnorm_alpha_masked = unnorm_alpha - 1000 * (1. - ctxt_mask)
        # else:
        #     unnorm_alpha_masked = unnorm_alpha
        # alpha = T.nnet.softmax(unnorm_alpha_masked) # bt_sz x T
        # attended_ctxt = T.batched_dot(alpha.dimshuffle((0,'x',1)), ctxt)[:,0,:] # bt_sz x ctxt_dim

        if self.implementation == 2:
            matrix_x = K.dot(inputs * dp_mask[0], self.kernel)
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, self.bias)
            matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
                                 self.recurrent_kernel[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            x_h = matrix_x[:, 2 * self.units:]
            recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                                self.recurrent_kernel[:, 2 * self.units:])
            hh = self.activation(x_h + recurrent_h)
        else:
            if self.implementation == 0:
                x_z = inputs[:, :self.units]
                x_r = inputs[:, self.units: 2 * self.units]
                x_h = inputs[:, 2 * self.units:]
            elif self.implementation == 1:
                x_z = K.dot(inputs * dp_mask[0], self.kernel_z)
                x_r = K.dot(inputs * dp_mask[1], self.kernel_r)
                x_h = K.dot(inputs * dp_mask[2], self.kernel_h)
                if self.use_bias:
                    x_z = K.bias_add(x_z, self.bias_z)
                    x_r = K.bias_add(x_r, self.bias_r)
                    x_h = K.bias_add(x_h, self.bias_h)
            else:
                raise ValueError('Unknown `implementation` mode.')
            z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                      self.recurrent_kernel_z)
                                              + K.dot(attended_ctxt, 
                                                      self.context_kernel_z))
            r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                      self.recurrent_kernel_r)
                                              + K.dot(attended_ctxt, 
                                                      self.context_kernel_r))
            hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2],
                                             self.recurrent_kernel_h)
                                     + K.dot(attended_ctxt, 
                                             self.context_kernel_h))
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, attended_ctxt]

    def get_config(self):
        config = {'attn_dense_dim': self.attn_dense_dim}
        base_config = super(AttentionDecoderGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return [(input_shape[0][0], input_shape[0][1], self.units), (input_shape[0][0], input_shape[0][1], self.ctxt_dim)]
        else:
            return (input_shape[0][0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return [mask[0], mask[0]]
        else:
            return None