from abc import ABC, abstractmethod

import numpy as np
import optuna
from optuna import Trial, Study
from optuna.trial import FixedTrial


class Estimator(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray):
        return NotImplemented


class CVOptimizer(ABC):
    def __init__(self, study: Study, is_one_cv=False, prune=False):
        # self.X = X
        # self.y = y
        self.study = study
        self.is_one_cv = is_one_cv
        self.prune = prune

    @abstractmethod
    def fit(self, trial: Trial, X_train, y_train, X_val, y_val, step) -> Estimator:
        return NotImplemented

    @abstractmethod
    def loss_fn(self, y_true, y_pred, trial: Trial):
        return NotImplemented

    @abstractmethod
    def split(self, X, y):
        return NotImplemented

    @abstractmethod
    def make_xy(self, trial: Trial):
        return NotImplemented

    def on_after_trial(self, trial: Trial, cv_models, cv_preds, loss_val):
        pass

    def best_models(self, return_preds=False):
        fixed_trial = FixedTrial(self.study.best_params)
        loss, models, cv_preds = self.objective(fixed_trial, return_model=True)

        if return_preds:
            return models, cv_preds
        return models

    def objective(self, trial: Trial, return_model=False):
        X, y = self.make_xy(trial)
        # X, y = self.X, self.y

        loss_cv_train = []

        cv_preds = np.zeros_like(y)
        cv_preds.fill(np.nan)

        models = []

        for step, (train_idx, val_idx) in enumerate(self.split(X, y)):
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            model = self.fit(trial, X_train, y_train, X_val, y_val, step)

            y_train_preds = model.predict(X_train)
            y_val_preds = model.predict(X_val)
            # from IPython.core.debugger import Pdb;
            # Pdb().set_trace()
            cv_preds[val_idx] = y_val_preds

            mask_done = ~np.isnan(cv_preds)
            intermediate_loss_train = self.loss_fn(y_train, y_train_preds, trial)
            intermediate_loss_val = self.loss_fn(y[mask_done], cv_preds[mask_done], trial)

            loss_cv_train.append(intermediate_loss_train)

            trial.report(intermediate_loss_val, step)
            if self.prune and trial.should_prune(step):
                raise optuna.structs.TrialPruned()

            models.append(model)

            if self.is_one_cv:
                break

        mask_done = ~np.isnan(cv_preds)
        loss_train = float(np.mean(loss_cv_train))
        loss_val = float(self.loss_fn(y[mask_done], cv_preds[mask_done], trial))

        trial.set_user_attr('train_loss', loss_train)
        trial.set_user_attr('val_loss', loss_val)
        trial.set_user_attr('is_one_cv', int(self.is_one_cv))

        self.on_after_trial(trial, models, cv_preds, loss_val)

        if return_model:
            return loss_val, models, cv_preds

        return loss_val

    def optimize(self, n_trials=100, n_jobs=1):
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
