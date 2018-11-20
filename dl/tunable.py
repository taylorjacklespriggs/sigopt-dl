class Tunable(object):
  def get_tunables(self):
    raise NotImplementedError(f'{type(self)} does not implement get_tunables')

  def get_value(self, context):
    raise NotImplementedError(f'{type(self)} does not implement get_value')
