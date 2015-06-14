import numpy as np
import theano
import theano.tensor as T

theano.config.on_unused_input = 'ignore'

from theanify import theanify, Theanifiable

class LSTM(Theanifiable):

    def __init__(self, n_input, n_hidden, num_layers=2, use_forget_gate=True):
        super(LSTM, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.use_forget_gate = use_forget_gate

        assert self.num_layers >= 1

        self.Wi = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wi')
        self.Ui = theano.shared(np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5, name='Ui')
        self.bi = theano.shared(np.random.rand(self.num_layers, self.n_hidden) - 0.5, name='bi')

        self.Whi = theano.shared(np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5, name='Whi')

        self.Wf = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wf')
        self.Uf = theano.shared(np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5, name='Uf')
        self.bf = theano.shared(np.random.rand(self.num_layers, self.n_hidden) - 0.5, name='bf')

        self.Whf = theano.shared(np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5, name='Whf')

        self.Wc = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wc')
        self.Uc = theano.shared(np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5, name='Uc')
        self.bc = theano.shared(np.random.rand(self.num_layers, self.n_hidden) - 0.5, name='bc')

        self.Whc = theano.shared(np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5, name='Whc')

        self.Wo = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5, name='Wo')
        self.Vo = theano.shared(np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5, name='Vo')
        self.Uo = theano.shared(np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5, name='Uo')
        self.bo = theano.shared(np.random.rand(self.num_layers, self.n_hidden) - 0.5, name='bo')

        self.Who = theano.shared(np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5, name='Who')

    def forward_with_weights(self, X, previous_hidden, previous_state, Wi, Ui, bi, Wf, Uf, bf, Wc, Uc, bc, Wo, Vo, Uo, bo):
        input_gate      = T.nnet.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)
        candidate_state = T.tanh(T.dot(X, Wc) + T.dot(previous_hidden, Uc) + bc)

        if self.use_forget_gate:
            forget_gate     = T.nnet.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + bf)
            state           = candidate_state * input_gate + previous_state * forget_gate
        else:
            state           = candidate_state * input_gate + previous_state * 0

        output_gate     = T.nnet.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) \
                                        + T.dot(state, Vo) + bo)
        output          = output_gate * T.tanh(state)
        return output, state

    @theanify(T.matrix('X'), T.tensor3('previous_hidden'), T.tensor3('previous_state'))
    def step(self, X, previous_hidden, previous_state):
        out, state = self.forward_with_weights(X, previous_hidden[:, 0, :], previous_state[:, 0, :],
                                               self.Wi, self.Ui[0], self.bi[0],
                                               self.Wf, self.Uf[0], self.bf[0],
                                               self.Wc, self.Uc[0], self.bc[0],
                                               self.Wo, self.Vo[0], self.Uo[0], self.bo[0])
        outs = [state]
        states = [state]
        for l in xrange(1, self.num_layers):
            out, state = self.forward_with_weights(out, previous_hidden[:, l, :], previous_state[:, l, :],
                                                self.Whi[l - 1], self.Ui[l], self.bi[l],
                                                self.Whf[l - 1], self.Uf[l], self.bf[l],
                                                self.Whc[l - 1], self.Uc[l], self.bc[l],
                                                self.Who[l - 1], self.Vo[l], self.Uo[l], self.bo[l])
            states.append(state)
            outs.append(out)
        return T.swapaxes(T.stack(*outs), 0, 1), T.swapaxes(T.stack(*states), 0, 1)

    def parameters(self):
        params = [self.Wi, self.Ui, self.bi, self.Wo, self.Vo, self.Uo, self.bo, self.Wc, self.Uc, self.bc]
        if self.num_layers > 1:
            params.extend([self.Whi, self.Who, self.Whc])
        if self.use_forget_gate:
            params.extend([self.Wf, self.Uf, self.bf])
            if self.num_layers > 1:
                params.extend([self.Whf])
        return params

if __name__ == "__main__":
    layers = 2
    O = 30
    B = 15
    D = 10
    lstm = LSTM(D, O, num_layers=layers).compile()
    X = np.ones((B, D))
    H = np.zeros((B, layers, O))
    S = np.zeros((B, layers, O))
