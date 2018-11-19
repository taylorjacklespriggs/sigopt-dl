from dl.param import Param

class BoundedParam(Param):
  def __init__(self, default_value, minimum, maximum):
    super().__init__(default_value)
    self.minimum = self.validate_type(minimum)
    self.maximum = self.validate_type(maximum)
    assert self.minimum < self.maximum
