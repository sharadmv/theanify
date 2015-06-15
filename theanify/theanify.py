import os
import cPickle as pickle
import sys
sys.setrecursionlimit(10000)

import theano
import theano.tensor as T

def theanify(*args, **kwargs):
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
        assert self.obj is not None, "Make sure you call the Theanifiable constructor"
        return self.f(self.obj, *args)

class Theanifiable(object):

    def __init__(self):
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                obj.set_instance(self)

    def compile(self, cache=None):
        cache_location = cache
        attrs = {}
        if cache and os.path.exists(cache):
            with open(cache, 'rb') as fp:
                cache = pickle.load(fp)
        else:
            cache = {}
        dirty = False
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                updates = None
                if obj.updates:
                    updates = getattr(self, obj.updates)(*obj.args)
                if name in cache:
                    compiled = cache[name]
                else:
                    compiled = theano.function(
                        obj.args,
                        obj(*obj.args),
                        updates=updates
                    )
                    cache[name] = compiled
                    dirty = True
                attrs[name] = compiled
            elif isinstance(obj, T.sharedvar.SharedVariable):
                attrs[name] = obj
            elif callable(obj) and type(obj) != type(all.__call__):
                attrs[name] = obj
            elif name[0] != "_":
                attrs[name] = obj
        if dirty and cache_location:
            with open(cache_location, 'wb') as fp:
                pickle.dump(cache, fp)
        return CompiledTheano(attrs)

class CompiledTheano(object):
    def __init__(self, funcs):
        for name, expr in funcs.items():
            setattr(self, name, expr)
