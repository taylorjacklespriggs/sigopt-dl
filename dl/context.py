from dl.param import Param

def get_full_path(path):
  return ':'.join(path)

class Root(object):
  def __init__(self):
    self.suggestion = None

  def create_context(self, name, tunable):
    return Context(self, [name], tunable)

  def get_assignment(self, path, default):
    return self.suggestion.assignments.get(get_full_path(path), default)

class Context(object):
  def __init__(self, root, path, tunable):
    self.root = root
    self.path = path
    self.tunable = tunable
    sub_tunables = tunable.get_tunables()
    if sub_tunables is None:
      self.sub_contexts = None
    else:
      self.sub_contexts = {
        name: Context(self.root, self.path + [name], sub_tunable)
        for name, sub_tunable
        in sub_tunables.items()
      }

  def __getitem__(self, item):
    return self.sub_contexts[item].get_value()

  def is_param(self):
    return isinstance(self.tunable, Param)

  def get_experiment_params(self):
    return [
      {
        'name': get_full_path(path),
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

  def get_assignment(self):
    return self.root.get_assignment(self.path, self.tunable.default_value)

  def get_value(self):
    return self.tunable.get_value(self)

  def get_args(self, tunable_args):
    return tuple(self[i] for i, _ in enumerate(tunable_args))

  def get_kwargs(self, tunable_kwargs):
    return {kw: self[kw] for kw in tunable_kwargs}
