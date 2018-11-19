from dl.param import Param

class Root(object):
  def __init__(self):
    self.current = True
    self.suggestion = None

  def is_current(self):
    return self.current

  def expire(self):
    self.current = False

  def get_full_path(self, path):
    return ':'.join(path)

  def create_context(self, name, tunable):
    return Context(self, [name], tunable)

  def get_assignment(self, path, default):
    return self.suggestion.assignments.get(self.get_full_path(path), default)

class Context(object):
  NO_VALUE = object()

  def __init__(self, root, path, tunable):
    self.root = root
    self.path = path
    self.tunable = tunable
    self.tunable.set_context(self)
    sub_tunables = tunable.get_tunables()
    if sub_tunables is None:
      self.sub_contexts = None
    else:
      self.sub_contexts = {
        name: Context(self.root, self.path + [name], sub_tunable)
        for name, sub_tunable
        in sub_tunables.items()
      }
    self.value = self.NO_VALUE

  def is_current(self):
    return self.root.is_current()

  def is_param(self):
    return isinstance(self.tunable, Param)

  def get_experiment_params(self):
    return [
      {
        'name': self.root.get_full_path(path),
        **param,
      }
      for path, param in self.get_experiment_param_body()
    ]

  def get_experiment_param_body(self):
    if self.is_param():
      yield self.path, self.tunable.get_param_body()
    if self.sub_contexts:
      for name, sub_context in self.sub_contexts.items():
        yield from sub_context.get_experiment_param_body()

  def apply_context(self):
    self.tunable.set_context(self)
    if self.sub_contexts:
      for context in self.sub_contexts.values():
        context.apply_context()

  def get_value(self):
    assert self.root.is_current(), 'Context is stale'
    if self.value is self.NO_VALUE:
      if self.is_param():
        self.value = self.root.get_assignment(self.path, self.tunable.default_value)
      else:
        self.apply_context()
        self.value = self.tunable.get_value()
    return self.value
