import numpy as np
import theano
import theano.tensor as T

theano.config.on_unused_input = 'ignore'

from theanify import theanify, Theanifiable

class CharacterGenerator(Theanifiable):

    def __init__(self, lstm, output):
        super(CharacterGenerator, self).__init__()
        self.lstm = lstm
        self.H = self.lstm.n_hidden
        self.output = output

        assert self.output.n_input == self.H, "Bad layer configuration"

        self.average_gradient = [theano.shared(p.get_value() * 0) for p in self.parameters()]
        self.average_rms = [theano.shared(p.get_value() * 0) for p in self.parameters()]
        self.parameter_update = [theano.shared(p.get_value() * 0) for p in self.parameters()]

    @theanify(T.tensor3('X'), T.tensor3('Y'))
    def loss(self, X, Y):
        return ((Y - self.forward(X)[-1])**2).sum()

    @theanify(T.tensor3('X'), T.tensor3('Y'))
    def gradients(self, X, Y, clip=5):
        return map(lambda x: x.clip(-clip, clip), T.grad(cost=self.loss(X, Y), wrt=self.parameters()))

    def sgd_updates(self, X, Y, learning_rate):
        return [(p, p - learning_rate * g) for p, g in zip(self.parameters(), self.gradients(X, Y))]

    @theanify(T.tensor3('X'), T.tensor3('Y'), T.dscalar('learning_rate'), updates="sgd_updates")
    def sgd(self, X, Y, learning_rate):
        return self.loss(X, Y)

    @theanify(T.tensor3('X'), T.tensor3('Y'), updates="rmsprop_updates")
    def rmsprop(self, X, Y):
        return self.loss(X, Y)

    def rmsprop_updates(self, X, Y):
        grads = self.gradients(X, Y)
        next_average_gradient = [0.95 * avg + 0.05 * g for g, avg in zip(grads, self.average_gradient)]
        next_rms = [0.95 * rms + 0.05 * (g ** 2) for g, rms in zip(grads, self.average_rms)]
        next_parameter = [0.9 * param_update - 1e-4 * g / T.sqrt(rms - avg ** 2 + 1e-4)
                          for g, avg, rms, param_update in zip(grads,
                                                               self.average_gradient,
                                                               self.average_rms,
                                                               self.parameter_update)]

        average_gradient_update = [(avg, next_avg) for avg, next_avg in zip(self.average_gradient,
                                                                            next_average_gradient)]
        rms_update = [(rms, rms2) for rms, rms2 in zip(self.average_rms,
                                                               next_rms)]
        next_parameter_update = [(param, param_update) for param, param_update in zip(self.parameter_update,
                                                                                      next_parameter)]

        updates = [(p, p + param_update) for p, param_update in zip(self.parameters(), next_parameter)]

        return updates + average_gradient_update + rms_update + next_parameter_update

    @theanify(T.tensor3('X'), T.iscalar('length'))
    def generate(self, X, length):
        L, N, D = X.shape

        hidden_layer, states, outputs = self.forward(X)

        H = self.lstm.n_hidden

        def step(previous_hidden, previous_state, previous_output):
            out = previous_output.argmax(axis=1)
            lstm_hidden, state = self.lstm.step(out[:, np.newaxis], previous_hidden, previous_state)
            final_output = self.output.forward(lstm_hidden)
            return lstm_hidden, state, final_output

        rval, _ = theano.scan(step,
                              outputs_info=[hidden_layer[-1, :, :],
                                            states[-1, :, :],
                                            outputs[-1, :, :]
                                           ],
                              n_steps=length)
        return rval[-1].argmax(axis=2)


    @theanify(T.tensor3('X'))
    def forward(self, X):
        L, N, D = X.shape
        H = self.lstm.n_hidden
        O = self.output.n_output

        def step(input, previous_hidden, previous_state, previous_output):
            lstm_hidden, state = self.lstm.step(input, previous_hidden, previous_state)
            final_output = self.output.forward(lstm_hidden)
            return lstm_hidden, state, final_output

        rval, _ = theano.scan(step,
                                sequences=[X],
                                outputs_info=[T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                           N,
                                                           H),
                                              T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                           N,
                                                           H),
                                              T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                           N,
                                                           O),
                                              ],
                                n_steps=L)
        return rval

    def parameters(self):
        return self.lstm.parameters() + self.output.parameters()

class Softmax(Theanifiable):

    def __init__(self, n_input, n_output):
        super(Softmax, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.Ws = theano.shared(np.random.random((self.n_input, self.n_output)))
        self.bs = theano.shared(np.random.random(self.n_output))

    @theanify(T.matrix('X'))
    def forward(self, X):
        return T.nnet.softmax(T.dot(X, self.Ws) + self.bs)

    def parameters(self):
        return [self.Ws, self.bs]
