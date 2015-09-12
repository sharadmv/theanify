import logging

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

    def compile(self):
        logging.info("Compiling <%s> object..." % self.__class__.__name__)
        attrs = {}
        attrs['_theanify_vars'] = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                logging.debug("Compiling <%s> method..." % name)
                updates = None
                if obj.updates:
                    updates = getattr(self, obj.updates)(*obj.args)
                var = obj(*obj.args)
                compiled = theano.function(
                    obj.args,
                    var,
                    updates=updates,
                    allow_input_downcast=True
                )
                attrs[name] = compiled
                attrs["_%s" % name] = obj
        for attr, obj in attrs.items():
            setattr(self, attr, obj)
        logging.info("Done compiling <%s> object." % self.__class__.__name__)
        return self
