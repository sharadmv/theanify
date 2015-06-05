import theano
import theano.tensor as T

def theano_optimize(*args, **kwargs):
    def func(f):
        return PreTheano(f, args, **kwargs)
    return func

class PreTheano(object):
    def __init__(self, f, args, updates=None):
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
                    updates = getattr(self, obj.updates)(*obj.args)
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
