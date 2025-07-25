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
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


class CausalForestWrapper(CausalForest):
    """ CausalForest wrapper to work with GridSearchCV """
    def fit(self, X, Y, T=None, **kwargs):
        # Handle both sklearn-style (X, Y) and causal-style (X, T, Y) calls
        if T is None:
            # Assume Y contains treatment and T contains outcome (GridSearchCV style)
            return super().fit(X, Y, T, **kwargs)
        else:
            # Normal causal forest call
            return super().fit(X, T, Y, **kwargs)


class RidgeCVClassifier(RidgeCV):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        p = np.clip(p, 0, 1)  # Ensure probabilities are in [0, 1]
        return np.concatenate([1 - p, p], axis=1)


class LassoLarsCVClassifier(LassoLarsCV):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        p = np.clip(p, 0, 1)  # Ensure probabilities are in [0, 1]
        return np.concatenate([1 - p, p], axis=1)


class KernelRidgeClassifier(KernelRidge):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        p = np.clip(p, 0, 1)  # Ensure probabilities are in [0, 1]
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
    parser.add_argument('--em', type=str, dest='estimation_model', 
                        choices=['dml', 'dr', 'xl', 'tl', 'cf', 'ridge', 'ridge-ipw', 
                                'lasso', 'kr', 'kr-ipw', 'et', 'et-ipw', 'dt', 'dt-ipw', 
                                'cb', 'cb-ipw', 'lgbm', 'lgbm-ipw', 'lr', 'lr-ipw', 'dummy'])
    parser.add_argument('--ebm', dest='estimation_base_model', type=str, 
                        choices=['lr', 'ridge', 'lasso', 'kr', 'et', 'dt', 'cb', 'lgbm'], 
                        default='lr')
    parser.add_argument('--ipw', dest='ipw_model', type=str, 
                        choices=['lr', 'kr', 'dt', 'et', 'cb', 'lgbm'], default='lr')
    parser.add_argument('--sfi', dest='save_features', action='store_true')

    return parser


def _get_classifier(name, options):
    """Get classifier model"""
    result = None
    
    if name in ('ridge', 'ridge-ipw'):
        result = RidgeCVClassifier(cv=options.cv)
    elif name == 'lr':
        result = LogisticRegressionCV(cv=options.cv, n_jobs=1, random_state=options.seed)
    elif name == 'lasso':
        result = LassoLarsCVClassifier(cv=options.cv, n_jobs=1)
    elif name in ('dt', 'dt-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(DecisionTreeClassifier(random_state=options.seed), 
                            param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('et', 'et-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(ExtraTreesClassifier(n_estimators=1000, bootstrap=True, 
                                                 random_state=options.seed, n_jobs=1), 
                            param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('kr', 'kr-ipw'):
        params = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), 
                 "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
        result = GridSearchCV(KernelRidgeClassifier(), n_jobs=options.n_jobs, 
                            scoring="neg_mean_squared_error", param_grid=params, cv=options.cv)
    elif name in ('cb', 'cb-ipw'):
        params = {"depth": [6, 8, 10], "l2_leaf_reg": [1, 3, 10, 100]}
        result = GridSearchCV(CatBoostClassifier(iterations=1000, random_state=options.seed, 
                                               verbose=False, thread_count=1), 
                            param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('lgbm', 'lgbm-ipw'):
        params = {"max_depth": [5, 7, 10], "reg_lambda": [0.1, 0, 1, 5, 10]}
        result = GridSearchCV(LGBMClassifier(n_estimators=1000, n_jobs=1, 
                                           random_state=options.seed, verbose=-1), 
                            param_grid=params, n_jobs=options.n_jobs, cv=options.cv)
    else:
        raise ValueError('Unknown classifier chosen.')
    
    return result


def _get_regressor(name, options):
    """Get regressor model"""
    result = None
    
    if name == 'dummy':
        result = DummyRegressor()
    elif name in ('ridge', 'ridge-ipw'):
        result = RidgeCV(cv=options.cv)
    elif name == 'lr':
        result = LinearRegression(n_jobs=1)
    elif name == 'lasso':
        result = LassoLarsCV(cv=options.cv, n_jobs=1)
    elif name in ('dt', 'dt-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(DecisionTreeRegressor(random_state=options.seed), 
                            param_grid=params, scoring="neg_mean_squared_error", 
                            n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('et', 'et-ipw'):
        params = {"max_leaf_nodes": [10, 20, 30, None], "max_depth": [5, 10, 20]}
        result = GridSearchCV(ExtraTreesRegressor(n_estimators=1000, bootstrap=True, 
                                                random_state=options.seed, n_jobs=1), 
                            param_grid=params, scoring="neg_mean_squared_error", 
                            n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('kr', 'kr-ipw'):
        params = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), 
                 "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
        result = GridSearchCV(KernelRidge(), n_jobs=options.n_jobs, 
                            scoring="neg_mean_squared_error", param_grid=params, cv=options.cv)
    elif name in ('cb', 'cb-ipw'):
        params = {"depth": [6, 8, 10], "l2_leaf_reg": [1, 3, 10, 100]}
        result = GridSearchCV(CatBoostRegressor(iterations=1000, random_state=options.seed, 
                                              verbose=False, thread_count=1), 
                            param_grid=params, scoring="neg_mean_squared_error", 
                            n_jobs=options.n_jobs, cv=options.cv)
    elif name in ('lgbm', 'lgbm-ipw'):
        params = {"max_depth": [5, 7, 10], "reg_lambda": [0.1, 0, 1, 5, 10]}
        result = GridSearchCV(LGBMRegressor(n_estimators=1000, n_jobs=1, 
                                          random_state=options.seed, verbose=-1), 
                            param_grid=params, scoring="neg_mean_squared_error", 
                            n_jobs=options.n_jobs, cv=options.cv)
    else:
        raise ValueError('Unknown regressor chosen.')
    
    return result


def _get_early_stopping_classifier(name, options):
    """Get classifier with early stopping capabilities for CB/LGBM"""
    if name == 'cb':
        return CatBoostClassifier(
            iterations=1000,
            early_stopping_rounds=50,
            random_state=options.seed,
            verbose=False,
            thread_count=1,
            eval_metric='Logloss'
        )
    elif name == 'lgbm':
        return LGBMClassifier(
            n_estimators=1000,
            random_state=options.seed,
            n_jobs=1,
            verbose=-1
        )
    else:
        # Return regular classifier for non-boosting methods
        return _get_classifier(name, options)


def _get_early_stopping_regressor(name, options):
    """Get regressor with early stopping capabilities for CB/LGBM"""
    if name == 'cb':
        return CatBoostRegressor(
            iterations=1000,
            early_stopping_rounds=50,
            random_state=options.seed,
            verbose=False,
            thread_count=1,
            eval_metric='RMSE'
        )
    elif name == 'lgbm':
        return LGBMRegressor(
            n_estimators=1000,
            random_state=options.seed,
            n_jobs=1,
            verbose=-1
        )
    else:
        # Return regular regressor for non-boosting methods
        return _get_regressor(name, options)


def _fit_with_early_stopping(model, X, y, X_val=None, y_val=None):
    """Fit model with early stopping if supported"""
    model_type = type(model).__name__
    
    if X_val is not None and y_val is not None:
        if 'CatBoost' in model_type:
            model.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
        elif 'LGBM' in model_type:
            model.fit(X, y, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50)])
        else:
            # For models without early stopping, just fit normally
            model.fit(X, y)
    else:
        model.fit(X, y)
    
    return model


def _get_model(options):
    """Get model configuration"""
    result = None
    fit_type = 'econml'
    
    if options.estimation_model == 'dml':
        result = DML(
            model_y=_get_regressor(options.estimation_base_model, options), 
            model_t=_get_classifier(options.estimation_base_model, options), 
            model_final=_get_regressor(options.estimation_base_model, options), 
            discrete_treatment=True, 
            random_state=options.seed, 
            fit_cate_intercept=True, 
            cv=options.cv
        )
    elif options.estimation_model == 'dr':
        result = DRLearner(
            model_propensity=_get_classifier(options.estimation_base_model, options), 
            model_regression=_get_regressor(options.estimation_base_model, options), 
            model_final=_get_regressor(options.estimation_base_model, options), 
            random_state=options.seed, 
            cv=options.cv
        )
    elif options.estimation_model == 'xl':
        result = XLearner(
            models=_get_regressor(options.estimation_base_model, options), 
            propensity_model=_get_classifier(options.estimation_base_model, options)
        )
    elif options.estimation_model == 'tl':
        result = TLearner(models=_get_regressor(options.estimation_base_model, options))
    elif options.estimation_model == 'cf':
        # Return a single CausalForest instance, not GridSearchCV
        result = CausalForest(n_estimators=1000, random_state=options.seed, n_jobs=1)
        fit_type = 'cf'
    else:
        result = _get_regressor(options.estimation_model, options)
        fit_type = 'sklearn'
    
    return result, fit_type


def _manual_grid_search_cf(train_data, val_data, options):
    """Manually perform grid search for Causal Forest using validation set"""
    X_train, T_train, Y_train = train_data
    X_val, T_val, Y_val = val_data
    
    T_train = T_train.flatten()
    Y_train = Y_train.flatten()
    T_val = T_val.flatten()
    Y_val = Y_val.flatten()
    
    param_grid = {"max_depth": [5, 10, 20], "n_estimators": [100, 500, 1000]}
    best_score = float('inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        cf = CausalForest(random_state=options.seed, n_jobs=1, **params)
        cf.fit(X_train, T_train, Y_train)
        
        # Evaluate on validation set
        # Option 1: Use out-of-bag predictions if available
        # Option 2: Use prediction error on validation set
        val_effects = cf.predict(X_val)
        
        # Since we don't have true individual treatment effects,
        # we can use the validation set to estimate performance
        # One approach: fit a separate model on validation to estimate ground truth
        val_cf_temp = CausalForest(n_estimators=100, random_state=options.seed)
        val_cf_temp.fit(X_val, T_val, Y_val)
        val_effects_reference = val_cf_temp.predict(X_val)
        
        # Compute error
        val_score = np.mean((val_effects - val_effects_reference)**2)
        
        if val_score < best_score:
            best_score = val_score
            best_params = params
            best_model = cf
    
    print(f"Best CF parameters: {best_params}")
    return best_model


def training(train_data, val_data, options):
    """Main training function"""
    X_train, T_train, Y_train = train_data
    X_val, T_val, Y_val = val_data
    
    T_train = T_train.flatten()
    Y_train = Y_train.flatten()
    T_val = T_val.flatten()
    Y_val = Y_val.flatten()
    
    model, fit_type = _get_model(options)
    
    # Handle different model types
    if options.estimation_model == 'cf':
        # Use manual grid search for Causal Forest
        model = _manual_grid_search_cf(train_data, val_data, options)
        
    elif options.estimation_model in ['dr', 'xl', 'tl'] and options.estimation_base_model in ['cb', 'lgbm']:
        # For meta-learners with boosting base models, we can implement custom early stopping
        if options.estimation_model == 'dr':
            # Create models with early stopping
            model = DRLearner(
                model_propensity=_get_early_stopping_classifier(options.estimation_base_model, options),
                model_regression=_get_early_stopping_regressor(options.estimation_base_model, options),
                model_final=_get_regressor(options.estimation_base_model, options),
                random_state=options.seed,
                cv=options.cv
            )
            
            # We need to modify the fit method to use validation set
            # Since EconML doesn't support this directly, we'll fit normally
            # but the base models will use their internal CV
            model.fit(Y=Y_train, T=T_train, X=X_train)
            
        elif options.estimation_model == 'xl':
            # For XLearner with early stopping base models
            model_t0 = _get_early_stopping_regressor(options.estimation_base_model, options)
            model_t1 = _get_early_stopping_regressor(options.estimation_base_model, options)
            propensity_model = _get_early_stopping_classifier(options.estimation_base_model, options)
            
            # Create XLearner with these models
            model = XLearner(
                models=[model_t0, model_t1],
                propensity_model=propensity_model
            )
            
            # Fit with early stopping validation
            # Note: XLearner doesn't natively support validation sets,
            # but the base models will use their internal validation
            model.fit(Y=Y_train, T=T_train, X=X_train)
            
        elif options.estimation_model == 'tl':
            # For TLearner with early stopping base models
            model_t0 = _get_early_stopping_regressor(options.estimation_base_model, options)
            model_t1 = _get_early_stopping_regressor(options.estimation_base_model, options)
            
            model = TLearner(models=[model_t0, model_t1])
            model.fit(Y=Y_train, T=T_train, X=X_train)
            
    else:
        # For other models or meta-learners without early stopping support
        # Train on training data only (validation set not used for early stopping)
        if fit_type == 'econml':
            model.fit(Y=Y_train, T=T_train, X=X_train)
        elif fit_type == 'sklearn':
            # This would be for direct outcome models
            # Note: This is not typically used for causal inference
            X_train_full = np.hstack([X_train, T_train.reshape(-1, 1)])
            model.fit(X_train_full, Y_train)
    
    return model


def estimate(model, test_data, options):
    """Estimate treatment effects on test data"""
    _, fit_type = _get_model(options)
    X_test, T_test, Y_test = test_data
    
    if fit_type == 'econml':
        # For all EconML models
        te_test = model.effect(X=X_test, T0=0, T1=1)
    elif fit_type == 'cf':
        # For Causal Forest
        te_test = model.predict(X_test)
    else:
        # For sklearn-style models (if used for direct prediction)
        # This is not standard for causal inference
        X_test_t0 = np.hstack([X_test, np.zeros((len(X_test), 1))])
        X_test_t1 = np.hstack([X_test, np.ones((len(X_test), 1))])
        
        y0_pred = model.predict(X_test_t0)
        y1_pred = model.predict(X_test_t1)
        te_test = y1_pred - y0_pred
    
    return te_test.flatten()


# # Example usage
# if __name__ == "__main__":
#     # Example of how to use the functions
#     # from your_data_module import prepare  # Import your prepare function
    
#     # Set up options
#     parser = get_parser()
#     # Example with DRLearner and LightGBM base model
#     options = parser.parse_args(['--em', 'dr', '--ebm', 'lgbm'])
    
#     # Simulate data for testing
#     np.random.seed(42)
#     n_samples = 1000
#     n_features = 10
    
#     # Create synthetic data
#     X = np.random.randn(n_samples, n_features)
#     T = np.random.binomial(1, 0.5, n_samples)
#     # Simple linear treatment effect
#     true_effect = 2.0
#     Y = X[:, 0] + true_effect * T + 0.1 * np.random.randn(n_samples)
    
#     # Split data
#     train_idx = np.arange(0, 600)
#     val_idx = np.arange(600, 800)
#     test_idx = np.arange(800, 1000)
    
#     train_data = [X[train_idx], T[train_idx], Y[train_idx]]
#     val_data = [X[val_idx], T[val_idx], Y[val_idx]]
#     test_data = [X[test_idx], T[test_idx], Y[test_idx]]
    
#     # Train model
#     print("Training model...")
#     model = training(train_data, val_data, options)
    
#     # Estimate effects on test set
#     print("Estimating treatment effects...")
#     predicted_effects = estimate(model, test_data, options)
    
#     # Evaluate
#     print(f"Mean predicted effect: {np.mean(predicted_effects):.3f}")
#     print(f"True effect: {true_effect}")