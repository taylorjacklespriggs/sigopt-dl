from dl.param import Param

class CategoricalParam(Param):
  validation_type = str

  def __init__(self, default_value, choices):
    super().__init__(default_value)
    self.choices = [self.validate_type(choice) for choice in choices]

  def get_param_body(self):
    return {
      'type': 'categorical',
      'categorical_values': self.choices,
    }
