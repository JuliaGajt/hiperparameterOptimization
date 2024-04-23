from optuna import create_study, samplers
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


class ModelTraining:

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

        # print(self.params)

        params = {}
        for key, val in self.params.items():
            low, up = self.params[key].interval(1)
            if isinstance(self.params[key], randint(0, 1).__class__):
                params.update({key: trial.suggest_int(key, low+1, up+1)})
            else:
                params.update({key: trial.suggest_uniform(key, low, up)})

        # print(params)

        # params = {
        #     # 'max_depth': trial.suggest_int('max_depth', 2, 7 - 1),
        #     # 'learning_rate': trial.suggest_uniform('learning_rate', 0.05, 0.5),
        #     'n_estimators': trial.suggest_int('n_estimators', 20, 200),
        #     'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
        #     'subsample': trial.suggest_uniform('subsample', 0.3, 1.0)
        # }

        # print(self.regressor_model(**params))
        # print(type(self.regressor_model(**params)))

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
        # print(self.study.best_params)
        # print(self.study.best_value)

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


# tuned_regressor = ModelTraining(X_train, y_train, 50, 'neg_mean_squared_error',
#                                 val_met, regressor_model=GradientBoostingRegressor)
# tuned_regressor.optimize()
# print(tuned_regressor.study.best_params)
# print(tuned_regressor.study.best_value)
# tuned_regressor = GradientBoostingRegressor(**tuned_regressor.study.best_params, loss='absolute_error',
#                                             criterion='squared_error', random_state=42)
# tuned_regressor.fit(X_train, y_train)
