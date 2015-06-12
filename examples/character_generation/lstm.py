import numpy as np
import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

class LSTM(Theanifiable):

    def __init__(self, n_input, n_hidden, use_forget_gate=True):
        super(LSTM, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.use_forget_gate = use_forget_gate

        self.Wi = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wi')
        self.Ui = theano.shared(np.random.rand(self.n_hidden, self.n_hidden) - 0.5, name='Ui')
        self.bi = theano.shared(np.random.rand(self.n_hidden) - 0.5, name='bi')

        if self.use_forget_gate:
            self.Wf = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wf')
            self.Uf = theano.shared(np.random.rand(self.n_hidden, self.n_hidden) - 0.5, name='Uf')
            self.bf = theano.shared(np.random.rand(self.n_hidden) - 0.5, name='bf')

        self.Wc = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wc')
        self.Uc = theano.shared(np.random.rand(self.n_hidden, self.n_hidden) - 0.5, name='Uc')
        self.bc = theano.shared(np.random.rand(self.n_hidden) - 0.5, name='bc')

        self.Wo = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wo')
        self.Vo = theano.shared(np.random.rand(self.n_hidden, self.n_hidden) - 0.5, name='Vo')
        self.Uo = theano.shared(np.random.rand(self.n_hidden, self.n_hidden) - 0.5, name='Uo')
        self.bo = theano.shared(np.random.rand(self.n_hidden) - 0.5, name='bo')

    @theanify(T.matrix('X'), T.matrix('previous_hidden'), T.matrix('previous_state'))
    def step(self, X, previous_hidden, previous_state):
        input_gate      = T.nnet.sigmoid(T.dot(X, self.Wi) + T.dot(previous_hidden, self.Ui) + self.bi)
        candidate_state = T.tanh(T.dot(X, self.Wc) + T.dot(previous_hidden, self.Uc) + self.bc)

        if self.use_forget_gate:
            forget_gate     = T.nnet.sigmoid(T.dot(X, self.Wf) + T.dot(previous_hidden, self.Uf) + self.bf)
            state           = candidate_state * input_gate + previous_state * forget_gate
        else:
            state           = candidate_state * input_gate + previous_state * 0

        output_gate     = T.nnet.sigmoid(T.dot(X, self.Wo) + T.dot(previous_hidden, self.Uo) \
                                         + T.dot(state, self.Vo) + self.bo)
        output          = output_gate * T.tanh(state)
        return output, state

    def parameters(self):
        params = [self.Wi, self.Ui, self.bi, self.Wo, self.Vo, self.Uo, self.bo]
        if self.use_forget_gate:
            params.extend([self.Wf, self.Uf, self.bf])
        return params
