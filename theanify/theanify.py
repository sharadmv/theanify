import logging

import theano

def theanify(*args, **kwargs):
    def func(f):
        return PreTheano(f, args, **kwargs)
    return func

class PreTheano(object):
    def __init__(self, f, args, updates=None, returns_updates=False):
        self.f = f
        self.args = args
        self.obj = None
        self.updates = updates
        self.returns_updates = returns_updates

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

    def compile_method(self, name, args=None, set_var=True):
        obj = getattr(self, name)
        args = args or obj.args
        if isinstance(obj, PreTheano):
            logging.debug("Compiling <%s> method..." % name)
            updates = None
            if obj.updates:
                updates = getattr(self, obj.updates)(*args)
            var = obj(*args)
            if obj.returns_updates:
                updates = var[1]
                var = var[0]
            compiled = theano.function(
                args,
                var,
                updates=updates,
                allow_input_downcast=True
            )
            if set_var:
                setattr(self, name, compiled)
                setattr(self, "_%s" % name, obj)
            return compiled

    def compile(self):
        logging.info("Compiling <%s> object..." % self.__class__.__name__)
        attrs = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, PreTheano):
                compiled = self.compile_method(name, set_var=False)
                attrs[name] = compiled
                attrs["_%s" % name] = obj

        for name, val in attrs.items():
            setattr(self, name, val)
        logging.info("Done compiling <%s> object." % self.__class__.__name__)
        return self
