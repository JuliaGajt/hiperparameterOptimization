import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import cross_val_score

from GeneticAlgorithm import _sample


class SimulatedAnnealing:
    def __init__(self, initial_solution, cv, pipeline, params, initial_temperature=100,
                 cooling_rate=0.95, min_temperature=0.1):
        self.initial_solution = initial_solution
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.cv = cv
        self.pipeline = pipeline
        self.params = params
        self.basic_params = self.pipeline['model'].get_params()
        self.best_params_ = {}
        self.best_estimator_ = self.pipeline['model']
        self.visited = []
        self.best_score_ = 0

    def objective_function(self, individual, X, y):
        fresh_pipeline = self.pipeline
        individual = dict(self.basic_params, **individual)
        regressor = (fresh_pipeline['model'])
        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**individual)))

        return cross_val_score(fresh_pipeline, X, y, cv=self.cv, scoring='neg_mean_squared_error').mean()

    def neighbor_generator(self, solution):
        # param1, = random.sample(sorted(self.params.keys()), 1)

        new_params = solution.copy()
        gen_num = 0

        while new_params in self.visited and gen_num < 10:

            param1, = random.sample(sorted(self.params.keys()), 1)

            gen_num += 1
            new_params = solution.copy()

            # print(new_params)

            if type(self.params[param1]) is not list:
                low, up = self.params[param1].interval(1)

                if isinstance(self.params[param1], randint(0, 1).__class__):
                    if new_params[param1] == up+1:
                        new_params[param1] = int(up)
                    elif new_params[param1] == low+1:
                        new_params[param1] = int(low + 2)
                    else:
                        new_params[param1] = int(new_params[param1] + np.random.choice([-1, 1]))

                elif isinstance(self.params[param1], uniform(0, 1).__class__) or isinstance(self.params[param1], loguniform(0, 1).__class__):
                    if new_params[param1] == up:
                        new_params[param1] = max(low, up - self.params[param1].rvs())
                    elif new_params[param1] == low:
                        new_params[param1] = min(up, low + self.params[param1].rvs())
                    else:
                        new_val = self.params[param1].rvs()
                        new_params[param1] = new_params[param1] + np.random.choice([-new_val, new_val])
                        if new_params[param1] > up or new_params[param1] < low:
                            new_params[param1] = self.params[param1].rvs()
            else:
                new_params[param1] = random.choice(self.params[param1])

        return new_params

    def run(self, X, y):

        update_iter = 3

        current_solution = self.initial_solution
        current_cost = self.objective_function(current_solution, X, y)
        best_solution = current_solution
        best_cost = current_cost
        temperature = self.initial_temperature
        scores = [best_cost]
        temperatures = [temperature]
        self.visited = [current_solution]
        iteration = 0

        while temperature > self.min_temperature:

            iteration += 1

            new_solution = self.neighbor_generator(current_solution)
            new_cost = self.objective_function(new_solution, X, y)
            delta_cost = new_cost - current_cost
            self.visited.append(new_solution)

            if delta_cost < 0 or uniform().rvs() < np.exp(-5*delta_cost / temperature):
                current_solution = new_solution
                current_cost = new_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            if iteration % update_iter == 0:
                temperature *= self.cooling_rate

            scores.append(best_cost)
            temperatures.append(temperature)

        print(f'max iters reached at temperature: {temperature}')
        return best_solution, best_cost

    def fit(self, X, y):

        self.best_params_, self.best_score_ = self.run(X, y)

        fresh_pipeline = self.pipeline
        best_estimator = dict(self.basic_params, **self.best_params_)
        regressor = (fresh_pipeline['model'])
        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**best_estimator)))

        self.best_estimator_ = fresh_pipeline
        self.pipeline = fresh_pipeline

        self.pipeline.fit(X, y)
        return self
