from dl.tunable import Tunable

class Constant(Tunable):
  def __init__(self, value):
    self._value = value

  def get_tunables(self):
    return {}

  def get_value(self, context):
    return self._value
