class RepeatedTunable(Tunable):
  def __init__(self, tunable, count_param):
    self.tunable = tunable
    self.count_param = count_param

  def get_value(self):
    return [
      self.tunable.value
      for _ in range(self.count_param.value)
    ]
