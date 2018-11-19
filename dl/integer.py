from dl.bounded import BoundedParam

class IntegerParam(BoundedParam):
  validation_type = int

  def get_param_body(self):
    return {
      'type': 'int',
      'bounds': {
        'min': self.minimum,
        'max': self.maximum,
      },
    }
