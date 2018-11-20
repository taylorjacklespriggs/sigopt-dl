from contextlib import contextmanager
import sigopt

from dl.context import Root
from dl.tunable import Tunable

class Connection(object):
  def __init__(self, api_token=None):
    self.conn = sigopt.Connection(api_token)

  def create_experiment(self, name, budget, tunables, extra):
    return Experiment(self.conn, name, budget, tunables, extra)

class Experiment(object):
  def __init__(self, conn, name, budget, tunables, extra):
    self.tunables = tunables
    self.root_context = None
    self.contexts = None
    self.reset_context()
    parameters = [
      parameter
      for name, context in self.contexts.items()
      for parameter in context.get_experiment_params()
    ]
    self.conn = conn
    self.experiment = self.conn.experiments().create(
      name=name,
      observation_budget=budget,
      parameters=parameters,
      **extra,
    )

  def reset_context(self):
    self.root_context = Root()
    self.contexts = {
      name: self.root_context.create_context(name, tunable)
      for name, tunable in self.tunables.items()
    }

  def create_model(self, fcn):
    self.reset_context()
    suggestion = self.conn.experiments(self.experiment.id).suggestions().create()
    self.root_context.suggestion = suggestion
    value = None
    try:
      value = fcn({name: context.get_value() for name, context in self.contexts.items()}, suggestion)
    except Exception as e:
      print(e)
      pass
    if value is None:
      self.conn.experiments(self.experiment.id).observations().create(
        failed=True,
        suggestion=suggestion.id,
      )
    else:
      self.conn.experiments(self.experiment.id).observations().create(
        value=value,
        suggestion=suggestion.id,
      )

  def check_experiment_progress(self):
    self.experiment = self.conn.experiments(self.experiment.id).fetch()
    return self.experiment.progress.observation_budget_consumed < self.experiment.observation_budget

  def loop(self, fcn):
    while self.check_experiment_progress():
      self.create_model(fcn)
