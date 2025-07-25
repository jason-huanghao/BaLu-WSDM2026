import os
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from econml.dml import DML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner
from econml.grf import CausalForest
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LassoLarsCV, RidgeCV, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier


class CausalForestWrapper(CausalForest):
    """ CausalForest doesn't work with GridSearchCV because CF.fit expects (X, T, Y) params,
    but GS passes (X, Y, T). This overwrite is to fix this.
    """
    def fit(self, X, Y, T, **kwargs):
        return super().fit(X, T, Y, **kwargs)


class RidgeCVClassifier(RidgeCV):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)


class LassoLarsCVClassifier(LassoLarsCV):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)


class KernelRidgeClassifier(KernelRidge):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)


def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dtype', type=str, choices=['ihdp', 'jobs', 'news', 'twins'])
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--sr', dest='save_results', action='store_true')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--scale_bin', action='store_true', default=False)
    parser.add_argument('--scale_y', action='store_true', default=False)
    parser.add_argument('--tbv', dest='transform_bin_vars', action='store_true')
    parser.add_argument('--ty', dest='transform_y', action='store_true')
    parser.add_argument('--cv', type=int, default=5)

    # Estimation
    parser.add_argument('--em', type=str, dest='estimation_model', choices=['dml', 'dr', 'xl', 'tl', 'cf', 'ridge', 'ridge-ipw', 'lasso', 'kr', 'kr-ipw', 'et', 'et-ipw', 'dt', 'dt-ipw', 'cb', 'cb-ipw', 'lgbm', 'lgbm-ipw', 'lr', 'lr-ipw', 'dummy'])
    parser.add_argument('--ebm', dest='estimation_base_model', type=str, choices=['lr', 'ridge', 'lasso', 'kr', 'et', 'dt', 'cb', 'lgbm'], default='lr')
    parser.add_argument('--ipw', dest='ipw_model', type=str, choices=['lr', 'kr', 'dt', 'et', 'cb', 'lgbm'], default='lr')
    parser.add_argument('--sfi', dest='save_features', action='store_true')

    return parser


def _get_classifier(name, options, use_validation=False):
    result = None
    if name in ('ridge', 'ridge-ipw'):
        cv_value = 2 if use_validation else options.cv  # Use 2-fold when validation set is provided
        result = RidgeCVClassifier(cv=cv_value)
    elif name == 'lr':
        cv_value = 2 if use_validation else options.cv
        result = LogisticRegressionCV(cv=cv_value, n_jobs=1, random_state=options.seed)
    elif name == 'lasso':
        cv_value = 2 if use_validation else options.cv
        result = LassoLarsCVClassifier(cv=cv_value, n_jobs=1)
    elif name in ('dt', 'dt-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(DecisionTreeClassifier(random_state=options.seed), param_grid=params, n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    elif name in ('et', 'et-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(ExtraTreesClassifier(n_estimators=1000, bootstrap=True, random_state=options.seed, n_jobs=1), param_grid=params, n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    elif name in ('kr', 'kr-ipw'):
        params = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
        result = GridSearchCV(KernelRidgeClassifier(), n_jobs=options.n_jobs, scoring="neg_mean_squared_error", param_grid=params, cv=2 if use_validation else options.cv)
    elif name in ('cb', 'cb-ipw'):
        params = {"depth": [6, 8, 10], "l2_leaf_reg": [1, 3, 10, 100]}
        result = GridSearchCV(CatBoostClassifier(iterations=1000, random_state=options.seed, verbose=False, thread_count=1), param_grid=params, n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    elif name in ('lgbm', 'lgbm-ipw'):
        params = {"max_depth": [5, 7, 10], "reg_lambda": [0.1, 0, 1, 5, 10]}
        result = GridSearchCV(LGBMClassifier(n_estimators=1000, n_jobs=1, random_state=options.seed, verbose=-1), param_grid=params, n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    else:
        raise ValueError('Unknown classifier chosen.')
    return result


def _get_regressor(name, options, use_validation=False):
    result = None
    if name == 'dummy':
        result = DummyRegressor()
    elif name in ('ridge', 'ridge-ipw'):
        cv_value = 2 if use_validation else options.cv
        result = RidgeCV(cv=cv_value)
    elif name == 'lr':
        result = LinearRegression(n_jobs=1)
    elif name == 'lasso':
        cv_value = 2 if use_validation else options.cv
        result = LassoLarsCV(cv=cv_value, n_jobs=1)
    elif name in ('dt', 'dt-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(DecisionTreeRegressor(random_state=options.seed), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    elif name in ('et', 'et-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(ExtraTreesRegressor(n_estimators=1000, bootstrap=True, random_state=options.seed, n_jobs=1), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    elif name in ('kr', 'kr-ipw'):
        params = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
        result = GridSearchCV(KernelRidge(), n_jobs=options.n_jobs, scoring="neg_mean_squared_error", param_grid=params, cv=2 if use_validation else options.cv)
    elif name in ('cb', 'cb-ipw'):
        params = {"depth": [6, 8, 10], "l2_leaf_reg": [1, 3, 10, 100]}
        result = GridSearchCV(CatBoostRegressor(iterations=1000, random_state=options.seed, verbose=False, thread_count=1), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    elif name in ('lgbm', 'lgbm-ipw'):
        params = {"max_depth": [5, 7, 10], "reg_lambda": [0.1, 0, 1, 5, 10]}
        result = GridSearchCV(LGBMRegressor(n_estimators=1000, n_jobs=1, random_state=options.seed, verbose=-1), param_grid=params, scoring="neg_mean_squared_error", n_jobs=options.n_jobs, cv=2 if use_validation else options.cv)
    else:
        raise ValueError('Unknown regressor chosen.')
    return result


def _create_train_val_split(train_data, val_data):
    """Create a PredefinedSplit for train/validation split"""
    from sklearn.model_selection import PredefinedSplit
    
    X_train, T_train, Y_train = train_data
    X_val, T_val, Y_val = val_data
    
    # Concatenate train and validation data
    X_combined = np.concatenate([X_train, X_val])
    T_combined = np.concatenate([T_train, T_val])
    Y_combined = np.concatenate([Y_train, Y_val])
    
    # Create test_fold array: -1 for training, 0 for validation
    test_fold = np.concatenate([
        np.full(len(X_train), -1),  # Training samples
        np.full(len(X_val), 0)      # Validation samples
    ])
    
    cv = PredefinedSplit(test_fold)
    
    return [X_combined, T_combined, Y_combined], cv


def _get_model(options, use_validation=False):
    result = None
    fit_type = 'econml'
    
    if options.estimation_model == 'dml':
        cv_value = 2 if use_validation else options.cv
        result = DML(
            model_y=_get_regressor(options.estimation_base_model, options, use_validation), 
            model_t=_get_classifier(options.estimation_base_model, options, use_validation), 
            model_final=_get_regressor(options.estimation_base_model, options, use_validation), 
            discrete_treatment=True, 
            random_state=options.seed, 
            fit_cate_intercept=True, 
            cv=cv_value
        )
    elif options.estimation_model == 'dr':
        cv_value = 2 if use_validation else options.cv
        result = DRLearner(
            model_propensity=_get_classifier(options.estimation_base_model, options, use_validation), 
            model_regression=_get_regressor(options.estimation_base_model, options, use_validation), 
            model_final=_get_regressor(options.estimation_base_model, options, use_validation), 
            random_state=options.seed, 
            cv=cv_value
        )
    elif options.estimation_model == 'xl':
        result = XLearner(
            models=_get_regressor(options.estimation_base_model, options, use_validation), 
            propensity_model=_get_classifier(options.estimation_base_model, options, use_validation)
        )
    elif options.estimation_model == 'tl':
        result = TLearner(models=_get_regressor(options.estimation_base_model, options, use_validation))
    elif options.estimation_model == 'cf':
        params = {"max_depth": [5, 10, 20]}
        cf = CausalForestWrapper(n_estimators=1000, random_state=options.seed, n_jobs=1)
        cv_value = 2 if use_validation else options.cv
        result = GridSearchCV(cf, param_grid=params, n_jobs=options.n_jobs, scoring='neg_mean_squared_error', cv=cv_value)
        fit_type = 'cf'
    else:
        result = _get_regressor(options.estimation_model, options, use_validation)
        fit_type = 'sklearn'
    return result, fit_type


def training(train_data, val_data, options):
    X_train, T_train, Y_train = train_data
    T_train = T_train.flatten()
    Y_train = Y_train.flatten()
    
    # For methods that support validation data, we'll use train/validation split
    # For others, we'll use the training data only
    if options.estimation_model in ['dr', 'tl', 'xl']:
        # These methods don't use CV internally, so we train on train_data only
        model, fit_type = _get_model(options, use_validation=False)
    else:
        # For methods that use GridSearchCV internally, create train/val split
        combined_data, cv = _create_train_val_split(train_data, val_data)
        model, fit_type = _get_model(options, use_validation=True)
        
        # Update the cv parameter for GridSearchCV-based methods
        if hasattr(model, 'cv'):
            model.cv = cv
    
    if fit_type == 'econml':
        if options.estimation_model in ['dr', 'tl', 'xl']:
            # Train only on training data
            model.fit(Y=Y_train, T=T_train, X=X_train)
        else:
            # Use combined data with predefined split
            X_combined, T_combined, Y_combined = combined_data
            model.fit(Y=Y_combined.flatten(), T=T_combined.flatten(), X=X_combined)
    elif fit_type == 'cf':
        if options.estimation_model == 'cf':
            # For Causal Forest with GridSearchCV
            X_combined, T_combined, Y_combined = combined_data
            model.fit(X=X_combined, T=T_combined.flatten(), y=Y_combined.flatten())
        else:
            model.fit(X=X_train, T=T_train, y=Y_train)
    
    return model


def estimate(model, test, options):
    _, fit_type = _get_model(options)
    X_test = test[0]

    if fit_type == 'econml':
        te_test = model.effect(X=X_test, T0=0, T1=1)
    elif fit_type == 'cf':
        te_test = model.predict(X_test)
    return te_test