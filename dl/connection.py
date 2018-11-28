from contextlib import contextmanager
import sigopt

from dl.context import Root
from dl.tunable import Tunable

class Connection(object):
  def __init__(self, api_token=None):
    self.conn = sigopt.Connection(api_token)

  def create_experiment(self, name, budget, tunable, evaluator):
    return Experiment(self.conn, name, budget, tunable, evaluator)

class Experiment(object):
  def __init__(self, conn, name, budget, tunable, evaluator):
    self.tunable = tunable
    self.evaluator = evaluator
    self.root_context = None
    self.context = None
    self.reset_context()
    parameters = list(self.context.get_experiment_params())
    self.conn = conn
    self.experiment = self.conn.experiments().create(
      name=name,
      observation_budget=budget,
      parameters=parameters,
    )

  def reset_context(self):
    self.root_context = Root()
    self.context = self.root_context.create_context('', self.tunable)

  def create_model(self):
    self.reset_context()
    suggestion = self.conn.experiments(self.experiment.id).suggestions().create()
    self.root_context.suggestion = suggestion
    value = None
    try:
      value = self.evaluator(self.context.get_value().__call__())
    except ValueError as e:
      print(e)
    except Exception as e:
      raise
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

  def loop(self):
    while self.check_experiment_progress():
      self.create_model()
