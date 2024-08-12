import random

from matplotlib import pyplot as plt
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def _sample(params, param):
    distribution = params[param]
    if type(distribution) is not list:
        return distribution.rvs()
    else:
        return random.choice(distribution)

class GeneticAlgorithm:
    def __init__(self, pipeline, params, cv=5, population_size=30, generations=30, mutation_rate=0.35):
        self.best_scores = []
        self.pipeline = pipeline
        self.params = params
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.cv = cv
        self.basic_params = self.pipeline['model'].get_params()
        self.best_params_ = {}
        self.best_estimator_ = self.pipeline['model']

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {param: _sample(self.params, param) for param in self.params.keys()}
            population.append(individual)
        return population

    def _evaluate_individual(self, individual, X, y):
        fresh_pipeline = self.pipeline
        individual = dict(self.basic_params, **individual)
        regressor = (fresh_pipeline['model'])
        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**individual)))

        return cross_val_score(fresh_pipeline, X, y, cv=self.cv, scoring='neg_mean_squared_error').mean()

    def _crossover(self, parent1, parent2):
        child = {}
        for param in self.params.keys():
            child[param] = random.choice([parent1[param], parent2[param]])
        return child

    def _mutate(self, individual):
        mutated_individual = individual.copy()
        for param, value in mutated_individual.items():
            if random.random() < self.mutation_rate:
                mutated_individual[param] = _sample(self.params, param)
        return mutated_individual

    def optimize(self, X, y):
        population = self._initialize_population()
        for gen in range(self.generations):
            print(f"iter: {gen}")

            # Evaluate individuals
            scores = [(self._evaluate_individual(individual, X, y), individual) for individual in population]
            scores.sort(key=lambda x: x[0])

            best_score = scores[0][0]
            self.best_scores.append(best_score)

            # Select top individuals for reproduction
            elite_size = int(0.1 * len(population))
            elite = [individual for _, individual in scores[:elite_size]]

            # Crossover
            children = []
            for _ in range(self.population_size - elite_size):
                parent1, parent2 = random.choices(population, k=2)
                child = self._crossover(parent1, parent2)
                children.append(child)

            # Mutation
            mutated_children = [self._mutate(child) for child in children]

            # Next generation
            population = elite + mutated_children

        best_params = scores[0][1]
        return best_params

    def fit(self, X, y):

        self.best_params_ = self.optimize(X, y)

        fresh_pipeline = self.pipeline
        best_estimator = dict(self.basic_params, **self.best_params_)
        regressor = (fresh_pipeline['model'])
        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**best_estimator)))

        self.best_estimator_ = fresh_pipeline
        self.pipeline = fresh_pipeline

        self.pipeline.fit(X, y)
        return self
