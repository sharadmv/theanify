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

    @theanify(T.tensor3('X'), T.tensor3('Y'), T.dscalar('learning_rate'), updates="sgd_updates")
    def adadelta(self, X, Y, learning_rate):
        return self.loss(X, Y)

    def adadelta_updates(self, X, Y, learning_rate):
        grads = self.gradients(X, y)

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                for rg2, g in zip(running_grads2, grads)]

        f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared')

        updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
                for zg, ru2, rg2 in zip(zipped_grads,
                                        running_up2,
                                        running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
        return [(p, p - learning_rate * g) for p, g in zip(self.parameters(), self.gradients(X, Y))]

    @theanify(T.tensor3('X'), T.iscalar('length'))
    def generate(self, X, length):
        L, N, D = X.shape

        hidden_layer, states, outputs = self.forward(X)

        H = self.lstm.n_hidden

        def step(previous_hidden, previous_state, previous_output):
            out = theano.printing.Print("Out:")(previous_output.argmax(axis=1))
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
