from dl.bounded import BoundedParam

class DoubleParam(BoundedParam):
  validation_type = (int, float)

  def get_param_body(self):
    return {
      'type': 'double',
      'bounds': {
        'min': self.minimum,
        'max': self.maximum,
      },
    }
