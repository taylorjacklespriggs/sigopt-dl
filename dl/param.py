from dl.tunable import Tunable

class Param(Tunable):
  def __init__(self, default_value):
    super().__init__()
    self.default_value = self.validate_type(default_value)

  def get_param_body(self):
    raise NotImplementedError(f'Param {type(self)} does not implement the get_param_body method')

  def get_tunables(self):
    return None

  def get_value(self, context):
    return context.get_assignment()

  @classmethod
  def validate_type(cls, value):
    assert isinstance(value, cls.validation_type)
    return value
