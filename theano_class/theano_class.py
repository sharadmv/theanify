import theano
import theano.tensor as T

def theano_optimize(vars, updates=None):
    def func(f):
        return PreTheano(f, vars, updates)
    return func

class PreTheano(object):
    def __init__(self, f, args, updates):
        self.f = f
        self.args = args
        self.obj = None
        self.updates = updates

    def set_instance(self, obj):
        self.obj = obj

    def __call__(self, *args):
        assert self.obj is not None
        return self.f(self.obj, *args)

class TheanoBase(object):

    def __init__(self):
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                obj.set_instance(self)

    def compile(self):
        wut = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                updates = None
                if obj.updates:
                    updates = obj.updates(self, *obj.args)
                wut[name] = theano.function(
                    obj.args,
                    obj(*obj.args),
                    updates=updates
                )
            elif isinstance(obj, T.sharedvar.SharedVariable):
                wut[name] = obj
            elif hasattr(obj, '__call__'):
                wut[name] = obj
        return CompiledTheano(wut)

class CompiledTheano(object):
    def __init__(self, funcs):
        for name, expr in funcs.items():
            setattr(self, name, expr)

if __name__ == "__main__":
    import theano.tensor as T
    class Foo(TheanoBase):

        @theano_optimize([
            T.dvector(),
            T.dvector()
        ])
        def sum(self, x, y):
            return x + y

        @theano_optimize([
            T.dvector(),
            T.dvector()
        ])
        def foo(self, x, y):
            return self.sum(x, y) + x + y


    f = Foo()
    w = f.compile()
