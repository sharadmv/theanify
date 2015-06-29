import numpy as np
import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

class LogisticRegression(Theanifiable):

    def __init__(self, n_input, n_output):
        super(LogisticRegression, self).__init__()
        self.n_input, self.n_output = n_input, n_output
        self.b = theano.shared(np.zeros(n_output))
        self.W = theano.shared(np.zeros((n_input, n_output)))

    @theanify(T.matrix('X'), T.ivector('y'))
    def negative_log_likelihood(self, X, y):
        return -T.mean(T.log(self.softmax(X))[T.arange(y.shape[0]), y])

    @theanify(T.matrix('X'))
    def softmax(self, X):
        return T.nnet.softmax(T.dot(X, self.W) + self.b)

    @theanify(T.matrix('X'))
    def predict(self, X):
        return T.argmax(self.softmax(X), axis=1)

    @theanify(T.matrix('X'), T.ivector('y'))
    def errors(self, X, y):
        y_pred = self.predict(X)
        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        return T.mean(T.neq(y_pred, y))

    @theanify(T.matrix('X'), T.ivector('y'), T.dscalar('learning_rate'))
    def gradients(self, X, y, learning_rate):
        cost = self.negative_log_likelihood(X, y)
        g_W, g_b = T.grad(cost=cost, wrt=[self.W, self.b])
        g_b = T.grad(cost=cost, wrt=self.b)
        return (self.W - learning_rate * g_W, self.b - learning_rate * g_b)

    def updates(self, X, y, learning_rate):
        g_W, g_b = self.gradients(X, y, learning_rate)
        return [(self.W, g_W), (self.b, g_b)]

    @theanify(T.matrix('X'), T.ivector('y'), T.dscalar('learning_rate'), updates="updates")
    def run(self, X, y, learning_rate):
        return self.negative_log_likelihood(X, y)


if __name__ == "__main__":
    from mldata import load
    X, y, Xtest, ytest = load('mnist', subsample=0.5)
    X = X.astype(theano.config.floatX)
    y = y.astype(np.int32)
    Xtest = Xtest.astype(theano.config.floatX)
    ytest = ytest.astype(np.int32)
    lr = LogisticRegression(784, 10).compile()
    raise Exception()

    learning_rate = 0.5
    iterations = 1000
    for i in xrange(iterations):
        print lr.run(X, y, learning_rate / (i + 1))
    print lr.errors(Xtest, ytest)
