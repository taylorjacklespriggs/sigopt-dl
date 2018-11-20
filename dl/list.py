from dl.tunable import Tunable

class ListTunable(Tunable):
  @staticmethod
  def get_tunable_name(i):
    return f'tunable{i}'

  def __init__(self, tunables):
    self.tunables = tunables

  def get_tunables(self):
    return {
      self.get_tunable_name(i): tunable
      for i, tunable in enumerate(self.tunables)
    }

  def get_value(self, context):
    return [
      context[self.get_tunable_name(i)]
      for i, _ in enumerate(self.tunables)
    ]
