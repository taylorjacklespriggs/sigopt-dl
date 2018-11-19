class Tunable(object):
  def __init__(self):
    self.context = None

  def set_context(self, context):
    self.context = context

  def get_tunables(self):
    raise NotImplementedError(f'{type(self)} does not implement get_tunables')

  def get_value(self):
    raise NotImplementedError(f'{type(self)} does not implement get_value')

  @property
  def value(self):
    assert self.context is not None, 'Context has not been set yet'
    assert self.context.is_current(), 'Context is stale'
    return self.context.get_value()
