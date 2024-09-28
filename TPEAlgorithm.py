from optuna import create_study, samplers
from scipy.stats import randint
from sklearn.model_selection import cross_val_score


class TPEAlgorithm:

    def __init__(self, pipeline, cv, params, X, y, cv_scoring_metric='neg_mean_squared_error',
                 n_trials=80, sampler_seed=1):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self._cv_scoring_metric = cv_scoring_metric
        self._cv = cv
        self.study = None
        self._sampler_seed = sampler_seed
        self.pipeline = pipeline
        self.params = params
        self.basic_params = self.pipeline['model'].get_params()
        self.best_params_ = {}
        self.best_estimator_ = self.pipeline['model']
        self.best_score_ = 0

    def _objective(self, trial):

        params = {}
        for key, val in self.params.items():
            if type(self.params[key]) is not list:
                low, up = self.params[key].interval(1)
                if isinstance(self.params[key], randint(0, 1).__class__):
                    params.update({key: trial.suggest_int(key, low+1, up+1)})
                else:
                    params.update({key: trial.suggest_uniform(key, low, up)})
            else:
                params.update({key: trial.suggest_categorical(key, val)})

        fresh_pipeline = self.pipeline
        params_for_model = dict(self.basic_params, **params)
        regressor = (fresh_pipeline['model'])
        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**params_for_model)))

        score = cross_val_score(fresh_pipeline,
                                self.X,
                                self.y,
                                scoring=self._cv_scoring_metric,
                                cv=self._cv).mean()
        return score

    def optimize(self):
        self.study = create_study(sampler=samplers.TPESampler(seed=self._sampler_seed),
                                  direction='maximize')
        self.study.optimize(self._objective, n_trials=self.n_trials)

    def fit(self, X, y):

        self.optimize()

        self.best_params_ = self.study.best_params
        fresh_pipeline = self.pipeline

        best_estimator = dict(self.basic_params, **self.best_params_)
        regressor = (fresh_pipeline['model'])
        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**best_estimator)))

        self.best_estimator_ = fresh_pipeline
        self.pipeline = fresh_pipeline
        self.best_score_ = self.study.best_value

        self.pipeline.fit(X, y)

        return self