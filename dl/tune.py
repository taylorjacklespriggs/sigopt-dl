from dl.tunable import Tunable

def tune(**kwargs):
  return TunableFunction(kwargs).set_function

class TunableFunction(Tunable):
  def __init__(self, kwargs, function=None):
    self.tunable_kwargs = kwargs
    self.function = function

  def get_tunables(self):
    return self.tunable_kwargs

  def set_function(self, function):
    self.function = function
    return self

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)

  def get_value(self, context):
    kwargs = context.get_kwargs(self.tunable_kwargs)
    return BoundFunction(self, kwargs)

class BoundFunction(object):
  def __init__(self, tunable_function, kwargs):
    self.tunable_function = tunable_function
    self.bound_kwargs = kwargs

  def __call__(self, *args, **kwargs):
    full_kwargs = self.bound_kwargs.copy()
    full_kwargs.update(kwargs)
    return self.tunable_function(*args, **full_kwargs)
