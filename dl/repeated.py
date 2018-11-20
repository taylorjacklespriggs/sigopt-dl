from dl.tunable import Tunable

class RepeatedTunable(Tunable):
  def __init__(self, tunable, count_param):
    self.tunable = tunable
    self.count_param = count_param

  def get_tunables(self):
    return {
      'tunable': self.tunable,
      'count': self.count_param,
    }

  def get_value(self, context):
    return [
      context['tunable']
      for _ in range(context['count'])
    ]
