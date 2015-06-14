import numpy as np
import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

class MLP(Theanifiable):

    def __init__(self, n_input, n_hidden, n_output, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.num_hidden_layers = num_hidden_layers

        assert self.num_hidden_layers >= 1

        self.Win = theano.shared(np.random.rand(self.n_input, self.n_hidden) - 0.5)
        self.bin = theano.shared(np.random.rand(self.n_hidden) - 0.5)

        self.Wh = theano.shared(np.random.rand(self.num_hidden_layers - 1, self.n_hidden, self.n_hidden) - 0.5)
        self.bh = theano.shared(np.random.rand(self.num_hidden_layers - 1, self.n_hidden) - 0.5)

        self.Wout = theano.shared(np.random.rand(self.n_hidden, self.n_output) - 0.5)
        self.bout = theano.shared(np.random.rand(self.n_output) - 0.5)

        self.activation = lambda x: T.switch(x<0, 0, x)


    @theanify(T.matrix('X'))
    def forward(self, X):
        out = self.activation(T.dot(X, self.Win) + self.bin)
        for l in xrange(self.num_hidden_layers - 1):
            out = self.hidden_forward(out, l)
        return self.softmax(out)

    @theanify(T.matrix('X'), T.iscalar())
    def hidden_forward(self, X, l):
        return self.activation(T.dot(X, self.Wh[l, :, :]) + self.bh[l, :])

    @theanify(T.matrix('X'), T.ivector('y'))
    def negative_log_likelihood(self, X, y):
        return -T.mean(T.log(self.forward(X))[T.arange(y.shape[0]), y])

    @theanify(T.matrix('X'))
    def predict(self, X):
        return T.argmax(self.forward(X), axis=1)

    @theanify(T.matrix('X'))
    def softmax(self, X):
        return T.nnet.softmax(T.dot(X, self.Wout) + self.bout)

    @theanify(T.matrix('X'), T.ivector('y'), T.dscalar('learning_rate'))
    def gradients(self, X, y, learning_rate):
        cost = self.negative_log_likelihood(X, y)
        params = self.get_params()
        gradients = T.grad(cost=cost, wrt=params)
        return gradients

    def updates(self, X, y, learning_rate):
        updates = []
        params = self.get_params()
        gradients = self.gradients(X, y, learning_rate)
        for param, gradient in zip(params, gradients):
            updates.append((param, param - learning_rate * gradient))
        return updates

    def get_params(self):
        params = [self.Win, self.bin, self.Wout, self.bout]
        if self.num_hidden_layers > 1:
            params.extend([self.Wh, self.bh])
        return params

    @theanify(T.matrix('X'), T.ivector('y'))
    def errors(self, X, y):
        y_pred = self.predict(X)
        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        return T.mean(T.neq(y_pred, y))

    @theanify(T.matrix('X'), T.ivector('y'), T.dscalar('learning_rate'), updates="updates")
    def run(self, X, y, learning_rate):
        return self.negative_log_likelihood(X, y)

if __name__ == "__main__":
    from mldata import load
    X, y, _, _ = load('mnist', small=1)
    X, y, Xtest, ytest = load('mnist', subsample=0.7)
    X /= 255.0
    X = X.astype(theano.config.floatX)
    y = y.astype(np.int32)
    Xtest = Xtest.astype(theano.config.floatX)
    Xtest /= 255.0
    ytest = ytest.astype(np.int32)
    mlp = MLP(784, 100, 10, num_hidden_layers=1).compile()

    iterations = 10000
    learning_rate = 20.0

    batch_size = 500
    for i in xrange(iterations):
        u = np.random.randint(X.shape[0] - batch_size)
        print mlp.run(X[u:u+batch_size, :], y[u:u+batch_size], learning_rate / (i + 1))
    print mlp.errors(Xtest, ytest)
