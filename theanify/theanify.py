import os
import logging
import cPickle as pickle
import sys
sys.setrecursionlimit(10000)

import theano

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

    def get_var(self):
        return self(*self.args)

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
        logging.info("Compiling <%s> object..." % self.__class__.__name__)
        cache_location = cache
        attrs = {}
        attrs['_theanify_vars'] = {}
        if cache and os.path.exists(cache):
            with open(cache, 'rb') as fp:
                cache = pickle.load(fp)
        else:
            cache = {}
        dirty = False
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                logging.debug("Compiling <%s> method..." % name)
                updates = None
                if obj.updates:
                    updates = getattr(self, obj.updates)(*obj.args)
                if name in cache:
                    logging.debug("Using cached copy...")
                    compiled = cache[name]
                else:
                    var = obj(*obj.args)
                    #attrs['_theanify_vars'][name] = var
                    compiled = theano.function(
                        obj.args,
                        var,
                        updates=updates
                    )
                    cache[name] = compiled
                    dirty = True
                attrs[name] = compiled
                attrs["_%s" % name] = obj
            #elif isinstance(obj, T.sharedvar.SharedVariable):
                #attrs[name] = obj
            #elif callable(obj) and type(obj) != type(all.__call__):
                #attrs[name] = obj
            #elif name[0] != "_":
                #attrs[name] = obj
        if dirty and cache_location:
            with open(cache_location, 'wb') as fp:
                pickle.dump(cache, fp)
        for attr, obj in attrs.items():
            setattr(self, attr, obj)
        logging.info("Done compiling <%s> object." % self.__class__.__name__)
        return self

class CompiledTheano(object):
    def __init__(self, funcs):
        for name, expr in funcs.items():
            setattr(self, name, expr)
