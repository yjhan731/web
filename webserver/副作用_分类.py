# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:30:02 2024

@author: h
"""


# filter all the warnings
import warnings
warnings.filterwarnings("ignore")

# data manipulation tools
import pandas as pd
import pandas
import numpy as np

# modeling tools
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier



import torch
import sys
sys.path.append('C:\\Users\\h\\AppData\\Roaming\\Python\\Python310\\site-packages')
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
pyo.init_notebook_mode()

# model metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
# data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter

# oversampling
from imblearn.over_sampling import SMOTE

# hyperparameter tuning
import optuna

# model interpretation
import shap
shap.initjs()

# miscellaneous
from pprint import pprint
from joblib import dump
from tqdm import tqdm
from typing import Union, Callable, Tuple, List
import os




#main_df = pd.read_csv(r'E://指标预测药物治疗//四个队列数据预处理//final.csv', encoding='gbk')
#main_df = pd.read_excel(r'E://指标预测药物治疗//March//marhc_dln(1).xlsx', sheet_name='Sheet1')
main_df = pd.read_excel(r'./webserver/队列5_categrate.xlsx', sheet_name='Sheet1')
display(main_df)
#val_df = main_df.iloc[:136, :]
#main_df = main_df.iloc[136:, :]

'''
index = ['Sex', 'Age', 'BMI', 'HbA1c', 'FBG', 'SBP', 'DBP', 'Heart rate', 'TC', 'HDL', 
         'HGB', 'WBC', 'RBC', 'PLT', 'ALT', 'AST', 'BUN', 'LDL',
         'CR', 'AEYW']
'''
index = ['Sex', 'Age', 'BMI', 'HbA1c', 'FBG', 'SBP', 'DBP', 'Heart rate', 'TC', 'TG', 'HDL', 
         'HGB', 'WBC', 'PLT', 'AST', 'BUN', 'CR', 'AEYW']



#### val_Df
#val_df = val_df[index]
# drop the index column

#val_df.isna().sum()
#val_df = val_df.dropna()
#val_df.isna().sum()
#val_df = val_df.reset_index(drop=True)



main_df = main_df[index]
# drop the index column

main_df.isna().sum()
main_df = main_df.dropna()
main_df.isna().sum()
main_df = main_df.reset_index(drop=True)


encoded_df = main_df
#encoded_df.to_excel(r'E://指标预测药物治疗//March//encode_df.xlsx', index=False)

def performance_baseline(dataframe_: pandas.core.frame.DataFrame, plot_title: str, target_name: str = False,
                         PCC: bool = False, weight: int = 1.35) -> None:
    """
        Calculates the performance baseline (weight multiplied by PCC or Proportion Chance Criterion) and plots the distribution of the target.

        Parameters
        ------------
        dataframe_ : pandas.core.frame.DataFrame
              the dataframe that contains the categorical features to encode
        plot_title : str
              the title of the plot
        target_name : str
              the name of the target to plot
        PCC : bool; default = False
              if True, calculates the PCC and prints out the results, if False, it will just plot the distribution
        weight : float; default = 1.35
                the specific weight multiplied by PCC

        Returns
        ------------
        None
    """
    if not target_name:
        state_counts = Counter(dataframe_)
        df_state = pd.DataFrame.from_dict(state_counts, orient="index")
    else:
        state_counts = Counter(dataframe_[target_name])
        df_state = pd.DataFrame.from_dict(state_counts, orient="index")

    _, axs = plt.subplots(figsize=(12, 6))

    for spine in ["top", "bottom", "left", "right"]:
        axs.spines[spine].set_visible(False)

    sns.barplot(x=df_state.index, y=df_state[0], palette=["#5DADE2", "#515A5A"], ec="k")
    plt.title(plot_title, fontweight="bold")

    if PCC:
        NUM = (df_state[0] / df_state[0].sum()) ** 2

        display(df_state)
        print(f"{weight} * Proportion Chance Criterion: {weight * 100 * NUM.sum()}")


performance_baseline(encoded_df, plot_title="Whole Data Class Distribution", target_name="AEYW", PCC=True)



def normalize_splits(
        X: pandas.core.frame.DataFrame,
        y: pandas.core.series.Series,
        test_size_: float,
        random_state_: int,
        continuous_features: list,
        categorical_features: list):

    SCALER_ = MinMaxScaler()

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X,
                                                        y,
                                                        test_size=test_size_,
                                                        random_state=random_state_,
                                                        stratify=y)
    smote = SMOTE(random_state=1)

    X_TRAIN_OVERSAMPLED, Y_TRAIN_OVERSAMPLED = smote.fit_resample(X_TRAIN, Y_TRAIN)

    SCALER_.fit(X_TRAIN_OVERSAMPLED[continuous_features])

    X_train_scaled = SCALER_.transform(X_TRAIN_OVERSAMPLED[continuous_features])
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=continuous_features)

    X_train_scaled = pd.concat([X_train_scaled, X_TRAIN_OVERSAMPLED[categorical_features].reset_index(drop=True)],
                               axis=1)

    X_test_scaled = SCALER_.transform(X_TEST[continuous_features])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=continuous_features)

    X_test_scaled = pd.concat([X_test_scaled, X_TEST[categorical_features].reset_index(drop=True)], axis=1)

    #Y_TEST_ENCODED = Y_TEST.map({"NEGATIVE": 0, "POSITIVE": 1})

    SCALED_SPLITS = (X_train_scaled, X_test_scaled, Y_TRAIN_OVERSAMPLED, Y_TEST)

    return SCALER_, SCALED_SPLITS



X = encoded_df.iloc[:, :-1]
y = encoded_df.iloc[:, -1]

#continuous_features = ['Sex', 'Age', 'BMI', 'HbA1c', 'FBG', 'SBP', 'DBP', 'Heart rate', ]
#categorical_features = ['HDL', 'HGB', 'WBC', 'RBC', 'PLT', 'ALT', 'TC', 'AST', 'BUN', 'LDL', 'CR']
continuous_features = ['Age', 'BMI', 'HbA1c', 'FBG', 'SBP', 'DBP', 'Heart rate']
categorical_features = ['Sex', 'HDL', 'HGB', 'WBC', 'PLT', 'TC', 'TG', 'AST', 'BUN', 'CR']



TRAIN_SCALER, SCALED_SPLITS = normalize_splits(X,
                                               y,
                                               test_size_= 0.1,
                                               random_state_= 1,
                                               continuous_features= continuous_features,
                                               categorical_features = categorical_features
                                               )

X_TRAIN_SCALED, X_TEST, Y_TRAIN, Y_TEST = SCALED_SPLITS
'''
x_val = val_df.iloc[:, :-1]
y_val = val_df.iloc[:, -1]
x_val1 = TRAIN_SCALER.transform(x_val[continuous_features])
x_val1 = pd.DataFrame(x_val1, columns=continuous_features)
x_val1 = pd.concat([x_val1, x_val[categorical_features].reset_index(drop=True)], axis=1)
'''





performance_baseline(Y_TRAIN, plot_title="Training Data Class Distribution", PCC = True)

# Splitting the Train Set for Validation Set
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_TRAIN_SCALED,
                                                                    Y_TRAIN,
                                                                    test_size=0.2,
                                                                    random_state=1,
                                                                   stratify = Y_TRAIN)

X_test_val, X_val, y_test_val, y_val = train_test_split(X_test_val,
                                                        y_test_val,
                                                        test_size=0.5,
                                                        random_state=1,
                                                        stratify = y_test_val)

def show_study_summary(study: optuna.study.Study) -> None:
    """
        Display a summary of the optimization study.

        Parameters
        ----------
        study : optuna.study.Study
            The optimization study to summarize.

        Returns
        -------
        None
            This function does not return any values. It displays information and plots.
    """
    print("\033[1mBest Hyperparameters")
    pprint(study.best_params)
    print()

    print("\033[1mAccuracy for the Best Hyperparameters")
    print(study.best_value)

    optimization_history_plot = optuna.visualization.plot_optimization_history(study)
    param_importances_plot = optuna.visualization.plot_param_importances(study)
    parallel_coordinate_plot = optuna.visualization.plot_parallel_coordinate(study)

    optimization_history_plot.update_layout({"height": 600})
    param_importances_plot.update_layout({"height": 600})
    parallel_coordinate_plot.update_layout({"height": 600})

    optimization_history_plot.show()
    param_importances_plot.show()
    parallel_coordinate_plot.show()


def logreg_objective(trial: optuna.study.Study) -> float:
    """
        Objective function for optimizing logistic regression hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single optimization trial.

        Returns
        ----------
        float
            F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "C": trial.suggest_float("C", 0.1, 1)
    }

    LOGREG = LogisticRegression(**PARAMS, max_iter=2000)

    LOGREG.fit(X_train_val, y_train_val)

    THRESHOLD = trial.suggest_float('threshold', 0.1, 1)

    YHAT = [1 if proba[1] > THRESHOLD else 0 for proba in LOGREG.predict_proba(X_test_val)]

    F1_SCORE_ = roc_auc_score(y_test_val, YHAT)

    return F1_SCORE_

LOGREG_STUDY = optuna.create_study(direction="maximize", study_name="logreg_tuning")

LOGREG_STUDY.optimize(logreg_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)
show_study_summary(LOGREG_STUDY)



def forest_objective(trial: optuna.study.Study)-> float:
    """
        Objective function for optimizing random forest hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single optimization trial.

        Returns
        ----------
        float
            F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "n_estimators": trial.suggest_int('n_estimators', 20, 150),
        "max_depth": trial.suggest_int('max_depth', 10, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 2]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    }

    RANDOM_FOREST = RandomForestClassifier(**PARAMS)

    RANDOM_FOREST.fit(X_train_val, y_train_val)

    YHAT = RANDOM_FOREST.predict(X_test_val)

    F1_SCORE_ = roc_auc_score(y_test_val, YHAT)

    return F1_SCORE_

RAND_FOREST_STUDY = optuna.create_study(direction="maximize", study_name="forest_tuning")

RAND_FOREST_STUDY.optimize(forest_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)
show_study_summary(RAND_FOREST_STUDY)


def adaboost_objective(trial: optuna.Trial) -> float:
    PARAMS = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "learning_rate": trial.suggest_float('learning_rate', 0.0001, 0.1),
        "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])
    }

    ADABOOST = AdaBoostClassifier(**PARAMS)

    ADABOOST.fit(X_train_val, y_train_val)

    YHAT = ADABOOST.predict(X_test_val)

    F1_SCORE_ = f1_score(y_test_val, YHAT)

    return F1_SCORE_

# 示例用法
adaboost_study = optuna.create_study(direction='maximize',  study_name="adaboost_tuning")
adaboost_study.optimize(adaboost_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)
show_study_summary(adaboost_study)


from sklearn.tree import DecisionTreeClassifier
def dt_objective(trial: optuna.Trial) -> float:
    """
    Objective function for optimizing Decision Tree hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        A single optimization trial.

    Returns
    ----------
    float
        F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "max_depth": trial.suggest_int('max_depth', 1, 30),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    }

    DECISION_TREE = DecisionTreeClassifier(**PARAMS)

    DECISION_TREE.fit(X_train_val, y_train_val)

    YHAT = DECISION_TREE.predict(X_test_val)

    F1_SCORE_ = f1_score(y_test_val, YHAT)

    return F1_SCORE_

# 示例用法
dt_study = optuna.create_study(direction='maximize',  study_name="dt_tuning")
dt_study.optimize(dt_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)

from sklearn.ensemble import ExtraTreesClassifier

def et_objective(trial: optuna.Trial) -> float:
    """
    Objective function for optimizing Extra Trees hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        A single optimization trial.

    Returns
    ----------
    float
        F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "max_depth": trial.suggest_int('max_depth', 10, 30),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }

    EXTRA_TREES = ExtraTreesClassifier(**PARAMS)

    EXTRA_TREES.fit(X_train_val, y_train_val)

    YHAT = EXTRA_TREES.predict(X_test_val)

    F1_SCORE_ = f1_score(y_test_val, YHAT)

    return F1_SCORE_

# 示例用法
et_study = optuna.create_study(direction='maximize')
et_study.optimize(et_objective, n_trials=150)


from sklearn.ensemble import GradientBoostingClassifier
def gbm_objective(trial: optuna.Trial) -> float:
    """
    Objective function for optimizing Gradient Boosting hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        A single optimization trial.

    Returns
    ----------
    float
        F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 1.0),
        "max_depth": trial.suggest_int('max_depth', 3, 30),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

    GBM = GradientBoostingClassifier(**PARAMS)

    GBM.fit(X_train_val, y_train_val)

    YHAT = GBM.predict(X_test_val)

    F1_SCORE_ = f1_score(y_test_val, YHAT)

    return F1_SCORE_

# 示例用法
gbm_study = optuna.create_study(direction='maximize')
gbm_study.optimize(gbm_objective, n_trials=150)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def knn_objective(trial: optuna.Trial) -> float:
    """
    Objective function for optimizing KNN hyperparameters.

    Parameters
    ----------
    trial : optuna.Trial
        A single optimization trial.

    Returns
    ----------
    float
        F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "n_neighbors": trial.suggest_int('n_neighbors', 1, 50),
        "weights": trial.suggest_categorical('weights', ['uniform', 'distance']),
        "algorithm": trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        "p": trial.suggest_int('p', 1, 5)  # For Minkowski metric, 1 is equivalent to Manhattan distance and 2 to Euclidean distance
    }

    KNN = KNeighborsClassifier(**PARAMS)

    # 使用管道进行数据标准化和模型训练
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNN)
    ])

    pipeline.fit(X_train_val, y_train_val)

    YHAT = pipeline.predict(X_test_val)

    F1_SCORE_ = f1_score(y_test_val, YHAT)

    return F1_SCORE_

# 示例用法
knn_study = optuna.create_study(direction='maximize')
knn_study.optimize(knn_objective, n_trials=150)



def SVC_objective(trial: optuna.study.Study) -> float:
    """
        Objective function for optimizing Support Vector Machine (SVM) hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single optimization trial.

        Returns
        ----------
        float
            F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "C": trial.suggest_float("C", 0.1, 1, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    }

    SVC_MODEL = SVC(**PARAMS)

    SVC_MODEL.fit(X_train_val, y_train_val)

    YHAT = SVC_MODEL.predict(X_test_val)

    F1_SCORE_ = roc_auc_score(y_test_val, YHAT)

    return F1_SCORE_

SVM_SVC_STUDY = optuna.create_study(direction="maximize", study_name="SVM_tuning")

SVM_SVC_STUDY.optimize(SVC_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)

show_study_summary(SVM_SVC_STUDY)


### xgboost

def XGB_objective(trial: optuna.study.Study) -> float:
    """
        Objective function for optimizing XGBoost hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single optimization trial.

        Returns
        ----------
        float
            F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 0.9),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    }

    XGB_model = xgb.XGBClassifier(**PARAMS)

    XGB_model.fit(X_train_val, y_train_val)

    YHAT = XGB_model.predict(X_test_val)

    F1_SCORE_ = roc_auc_score(y_test_val, YHAT)

    return F1_SCORE_

XGB_STUDY = optuna.create_study(direction="maximize", study_name="XGB_tuning")

XGB_STUDY.optimize(XGB_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)

show_study_summary(XGB_STUDY)


 # light gradient boosting
def lightgbm_objective(trial: optuna.study.Study) -> float:
    """
        Objective function for optimizing LightGBM hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single optimization trial.

        Returns
        ----------
        float
            F1 score obtained using the specified hyperparameters.
    """
    PARAMS = {
        "data_sample_strategy": "goss",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int('num_leaves', 10, 200, step=10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 4, 5]),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 0.9)
    }
    
    LGMB_MODEL = lgb.LGBMClassifier(**PARAMS)
    
    LGMB_MODEL.fit(X_train_val, y_train_val) 
    
    YHAT = LGMB_MODEL.predict(X_test_val)
    
    F1_SCORE_ = roc_auc_score(y_test_val, YHAT)
    
    return F1_SCORE_

LGBM_STUDY = optuna.create_study(direction="maximize", study_name="lgbm_tuning")

LGBM_STUDY.optimize(lightgbm_objective, n_trials=150, n_jobs=-1, show_progress_bar=True)

show_study_summary(LGBM_STUDY)



def test_model(
        models: list,
        params: dict,
        X_train: Union[np.array, pandas.core.frame.DataFrame],
        y_train: Union[np.array, pandas.core.frame.DataFrame],
        X_test: Union[np.array, pandas.core.frame.DataFrame],
        y_test: Union[np.array, pandas.core.frame.DataFrame]
    ) -> Tuple[Union[dict, pandas.core.frame.DataFrame]]:
    """
        Test multiple machine learning models on a given dataset and return evaluation scores.

        Parameters
        ----------
        models : list
            List of objets of machine learning model classes to be tested.
        params : dict
            Dictionary containing hyperparameters for each model.
        X_train : array-like or pd.DataFrame
            Training data.
        y_train : array-like or pd.Series
            Training labels.
        X_test : array-like or pd.DataFrame
            Testing data.
        y_test : array-like or pd.Series
            Testing labels.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - A dictionary containing trained models.
            - A pandas DataFrame with evaluation scores for each model.
    """

    model_testing_scores = pd.DataFrame()
    models_container = {}

    model_testing_scores["Model"] = []
    model_testing_scores["Accuracy"] = []
    model_testing_scores["Precision"] = []
    model_testing_scores["Recall"] = []
    model_testing_scores["F1-Score"] = []
    model_testing_scores["Specificity"] = []
    model_testing_scores["AUC"] = []
    model_testing_scores['fp'] = []
    model_testing_scores['tp'] = []
    #model_testing_scores['auc'] = []
    
    for model in tqdm(models):

        MODEL_NAME = model().__class__.__name__

        if MODEL_NAME == "LogisticRegression":
            test_model = model(tol=params[MODEL_NAME]["tol"],
                               C=params[MODEL_NAME]["C"],
                               max_iter=2000)

            test_model.fit(X_train, y_train)

            test_pred = [1 if proba[1] > params[MODEL_NAME]["threshold"] else 0 for proba in
                         test_model.predict_proba(X_test)]
            test_pred1 = test_model.predict_proba(X_val)[:, 1]
            TN, FP, _, _ = confusion_matrix(y_test, test_pred).ravel()

            TESTING_SPECIFICITY = TN / (TN + FP)
            
            fp1, tp1, _ = roc_curve(y_test, test_pred1)
            
            model_testing_scores = model_testing_scores._append({"Model": MODEL_NAME,
                                                                "Accuracy": accuracy_score(y_test, test_pred),
                                                                "Precision": precision_score(y_test, test_pred),
                                                                "Recall": recall_score(y_test, test_pred),
                                                                "F1-Score": f1_score(y_test, test_pred),
                                                                "Specificity": TESTING_SPECIFICITY,
                                                                "AUC": auc(fp1, tp1),
                                                                'fp':fp1,
                                                                'tp':tp1},
                                                               ignore_index=True)
        else:
            test_model = model(**params[MODEL_NAME])

            test_model.fit(X_train, y_train)

            test_pred = test_model.predict(X_test)
            test_pred1 = test_model.predict_proba(X_val)[:, 1]

            TN, FP, _, _ = confusion_matrix(y_test, test_pred).ravel()

            TESTING_SPECIFICITY = TN / (TN + FP)
            fp1, tp1, _ = roc_curve(y_test, test_pred1)
            
            model_testing_scores = model_testing_scores._append({"Model": MODEL_NAME,
                                                                "Accuracy": accuracy_score(y_test, test_pred),
                                                                "Precision": precision_score(y_test, test_pred),
                                                                "Recall": recall_score(y_test, test_pred),
                                                                "F1-Score": f1_score(y_test, test_pred),
                                                                "Specificity": TESTING_SPECIFICITY,
                                                                "AUC": auc(fp1, tp1),
                                                                'fp':fp1,
                                                                'tp':tp1},
                                                               ignore_index=True)

        models_container[MODEL_NAME] = test_model

    return models_container, model_testing_scores





LIST_OF_MODELS = [LogisticRegression, RandomForestClassifier, SVC, xgb.XGBClassifier, lgb.LGBMClassifier, AdaBoostClassifier,
                  DecisionTreeClassifier, ExtraTreesClassifier, GradientBoostingClassifier, KNeighborsClassifier]

show_study_summary(LOGREG_STUDY)
show_study_summary(RAND_FOREST_STUDY)
show_study_summary(SVM_SVC_STUDY)
show_study_summary(XGB_STUDY)
show_study_summary(LGBM_STUDY)
show_study_summary(adaboost_study)
show_study_summary(dt_study)
show_study_summary(et_study)
show_study_summary(gbm_study)
show_study_summary(knn_study)
###  最优
MODELS_PARAMETERS = {"LogisticRegression":{'C': 0.5221346501542409,
                     'threshold': 0.40596597398724976,
                     'tol': 0.00033160234235611895},
                     
                    "RandomForestClassifier":{'criterion': 'gini', 'max_depth': 16, 'max_features': 2, 'n_estimators': 78},
                     
                    "SVC":{'C': 0.4684387046307116, 'kernel': 'poly', 'probability':True},
                     
                    "XGBClassifier":{'learning_rate': 0.07070875105227867,
                     'max_depth': 11,
                     'n_estimators': 750,
                     'reg_alpha': 0.6833947294369151},
                     
                     "LGBMClassifier": {"data_sample_strategy": "goss",
                                        "verbosity": -1,
                                        'boosting_type': 'dart',
                                         'learning_rate': 0.07418862911949028,
                                         'max_depth': -1,
                                         'n_estimators': 150,
                                         'num_leaves': 160,
                                         'reg_alpha': 0.8488483898906658},
                     "AdaBoostClassifier": {'algorithm': 'SAMME.R',
                      'learning_rate': 0.015812789775497757,
                      'n_estimators': 140},
                     "DecisionTreeClassifier": {'criterion': 'entropy',
                      'max_depth': 28,
                      'max_features': 'sqrt',
                      'min_samples_leaf': 3,
                      'min_samples_split': 15},
                     
                     "ExtraTreesClassifier": {'criterion': 'gini',
                      'max_depth': 30,
                      'max_features': 'log2',
                      'min_samples_leaf': 1,
                      'min_samples_split': 4,
                      'n_estimators': 232},
                     
                     "GradientBoostingClassifier": {'learning_rate': 0.26844273070788055,
                      'max_depth': 23,
                      'max_features': 'sqrt',
                      'min_samples_leaf': 9,
                      'min_samples_split': 17,
                      'n_estimators': 117},
                     
                     "KNeighborsClassifier": {'algorithm': 'kd_tree', 'n_neighbors': 8, 'p': 1, 'weights': 'distance'}
                    }







MODEL_CONTAINER, TESTING_SCORES = test_model(LIST_OF_MODELS, MODELS_PARAMETERS, X_train_val, y_train_val, X_val, y_val)
X_TEST
Y_TEST.sum()
TESTING_SCORES

# 创建 SHAP 解释器
p = MODELS_PARAMETERS['GradientBoostingClassifier']
p = MODELS_PARAMETERS['RandomForestClassifier']
p = MODELS_PARAMETERS['LGBMClassifier']
model = ExtraTreesClassifier(**p)
model = RandomForestClassifier(**p)
model = lgb.LGBMClassifier(**p)

from sklearn.ensemble import GradientBoostingClassifier
p = MODELS_PARAMETERS['GradientBoostingClassifier']
model = GradientBoostingClassifier(**p)
model.fit(X_train_val, y_train_val)

explainer = shap.Explainer(model)
shap_values = explainer(X_val)

# 绘制 SHAP 图
shap.summary_plot(shap_values, X_val)

## 特征重要性
shap.summary_plot(shap_values, X_val, 
                  plot_type="bar")


####  绘制auc和feature importance 相关图

from sklearn.model_selection import StratifiedKFold

#explainer = shap.Explainer(model)
#shap_values = explainer(X_test_val)

# 获取特征重要性
importances = np.abs(shap_values.values).mean(0)
indices = np.argsort(importances)[::-1]
feature_names_sorted = X_train_val.columns[indices]

# 打印特征重要性
for f in range(X_train_val.shape[1]):
    print(f"{f + 1}. feature {X_train_val.columns[indices[f]]} ({importances[indices[f]]})")

# 准备绘图的数据
feature_counts = list(range(1, len(indices) + 1))
mean_auc_scores = []
std_auc_scores = []

# 根据特征重要性选择不同数量的特征，并评估模型性能
for count in feature_counts:
    selected_features = X_train_val.columns[indices[:count]]
    X_train_selected = X_train_val[selected_features]
    X_test_selected = X_val[selected_features]
    
    # 进行五折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_index, test_index in kf.split(X_train_selected, y_train_val):
        X_train_fold, X_test_fold = X_train_selected.iloc[train_index], X_train_selected.iloc[test_index]
        y_train_fold, y_test_fold = y_train_val.iloc[train_index], y_train_val.iloc[test_index]
        
        #model = RandomForestClassifier(random_state=42)
        model = GradientBoostingClassifier(**p)
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
        auc = roc_auc_score(y_test_fold, y_pred_proba)
        auc_scores.append(auc)
    
    mean_auc_scores.append(np.mean(auc_scores))
    std_auc_scores.append(np.std(auc_scores))

###  选择11个特征的模型 作为web端模型
count = 11

['Sex', 'BUN', 'Heart rate', 'FBG', 'HDL', 'HbA1c', 'BMI', 'Age', 'TC',
       'SBP', 'DBP']

selected_features = X_train_val.columns[indices[:count]]
X_train_selected = X_train_val[selected_features]
X_test_selected = X_val[selected_features]
model = GradientBoostingClassifier(**p)
model.fit(X_train_selected, y_train_val)
y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)



### 保存模型以及实例测试
import joblib

joblib.dump(TRAIN_SCALER, r'./webserver/fzy_scaler.pkl')
joblib.dump(model, r'./webserver/fzy_model.pkl')

scaler_fzy = joblib.load(r'./webserver/fzy_scaler.pkl')
model_fzy = joblib.load(r'./webserver/fzy_model.pkl')


cate = {'Sex':2, 'BUN':1, 'HDL':1, 'TC':2}
cont = {'Age':45, 'BMI':27.2392, 'HbA1c':7.6, 'FBG':8.6, 'SBP':120, 'DBP':92, 'Heart rate':96}
std_fea = np.array([cont['Age'], cont['BMI'], cont['HbA1c'], cont['FBG'], cont['SBP'], cont['DBP'], cont['Heart rate'], ])
std_fea_reshape = std_fea.reshape(1, -1)
cont_scaled = scaler_fzy.transform(std_fea_reshape)
cont_scaled.shape
input_data = np.array([cate['Sex'], cate['BUN'], cont_scaled[0][3], cate['HDL'], cont_scaled[0][6], cont_scaled[0][1], 
                      cont_scaled[0][2], cont_scaled[0][0], cont_scaled[0][5], cont_scaled[0][4], cate['TC']])
input_data = input_data.reshape(1, -1)
y_pred_proba1 = model_fzy.predict_proba(input_data)[:, 1]
result = round(y_pred_proba1[0], 4)

#### 

mean_auc_scores = np.array(mean_auc_scores)
std_auc_scores = np.array(std_auc_scores)
auc_upper = mean_auc_scores + std_auc_scores / np.sqrt(5)
auc_lower = mean_auc_scores - std_auc_scores / np.sqrt(5)

fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制特征重要性
color = 'tab:blue'
#ax1.set_xlabel('Feature')
ax1.set_ylabel('Predictor Importance', color=color)
ax1.bar(feature_names_sorted, importances[indices], color=color, alpha=0.6, label='Predictor Importance')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(feature_names_sorted, rotation=90)

# 使用双 y 轴绘制AUC值及其误差区间
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Cumulative AUC', color=color)
ax2.plot(feature_names_sorted, mean_auc_scores, color=color, marker='o', label='Cumulative AUC')
ax2.fill_between(feature_names_sorted, auc_lower, auc_upper, color='pink', alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
#plt.title('Feature Importance and AUC (5-Fold Cross Validation)')
plt.show()



## 相关性图

import matplotlib.pyplot as plt

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['BUN']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('BUN')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('SBP(mmHg)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for SBP')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 
# 获取特征实际值和 SHAP 值
feature_values = X_test_val['FBG']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('FBG')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('FBG(mmol/L)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for FBG')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['Heart rate']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('Heart rate')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('Heart rate(times per minute)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for Heart rate')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['BMI']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('BMI')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('BMI(kg/m**2)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for BMI')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['TC']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('TC')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('TC(mmol/L)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for TC')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['HbA1c']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('HbA1c')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('HbA1c(%)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for HbA1c')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()


# 获取特征实际值和 SHAP 值
feature_values = X_test_val['Age']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('Age')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('Age')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for Age')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['ALT']  # 假设 'SBP' 是您感兴趣的特征

shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('ALT')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('ALT(normalcy/exceptions)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for ALT')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()

# 获取特征实际值和 SHAP 值
feature_values = X_test_val['TC']  # 假设 'SBP' 是您感兴趣的特征
shap_values_sbp = shap_values.values[:, X_test_val.columns.get_loc('TC')]  # 获取 'SBP' 对应的 SHAP 值

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, shap_values_sbp, alpha=0.5)
plt.xlabel('TC(normalcy/exceptions)')  # 横坐标为特征实际值
plt.ylabel('SHAP Value for TC')  # 纵坐标为 SHAP 值
#plt.title('Dependence Plot of SBP')  # 图形标题
plt.grid(True)  # 添加网格线
plt.show()


# 绘制结果柱状图
TESTING_SCORES1 = TESTING_SCORES.iloc[:,:7]
plot_df = pd.melt(TESTING_SCORES1, id_vars=["Model"])
plot_df.rename({"variable": "Metric", "value": "Score"}, axis=1, inplace=True)

# Create a horizontal bar plot using Seaborn
PLOT = sns.catplot(x="Score", y="Model", hue="Metric", data=plot_df, kind="bar", orient="h",
                   height=6, aspect=1.5, legend_out=False)

# Access the axes of the plot
ax = PLOT.axes[0, 0]

# Annotate the bars with their corresponding scores
for p in ax.patches:
    ax.annotate(f" {p.get_width(): .3f}", (p.get_x() + p.get_width(), (p.get_y() + 0.045) + p.get_height() / 2), ha="left")

# Set plot title, legend, and display the plot
plt.title("Model Evaluation Results")
plt.legend(loc=(1.05, 0.75))
plt.show()






####  绘制结果roc曲线

plt.figure(figsize=(10, 6))
plt.plot(TESTING_SCORES.iloc[0, 7], TESTING_SCORES.iloc[0, 8], label=f'{TESTING_SCORES.iloc[0, 0]}: AUC = {TESTING_SCORES.iloc[0, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[1, 7], TESTING_SCORES.iloc[1, 8], label=f'{TESTING_SCORES.iloc[1, 0]}: AUC = {TESTING_SCORES.iloc[1, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[2, 7], TESTING_SCORES.iloc[2, 8], label=f'{TESTING_SCORES.iloc[2, 0]}: AUC = {TESTING_SCORES.iloc[2, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[3, 7], TESTING_SCORES.iloc[3, 8], label=f'{TESTING_SCORES.iloc[3, 0]}: AUC = {TESTING_SCORES.iloc[3, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[4, 7], TESTING_SCORES.iloc[4, 8], label=f'{TESTING_SCORES.iloc[4, 0]}: AUC = {TESTING_SCORES.iloc[4, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[5, 7], TESTING_SCORES.iloc[5, 8], label=f'{TESTING_SCORES.iloc[5, 0]}: AUC = {TESTING_SCORES.iloc[5, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[6, 7], TESTING_SCORES.iloc[6, 8], label=f'{TESTING_SCORES.iloc[6, 0]}: AUC = {TESTING_SCORES.iloc[6, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[7, 7], TESTING_SCORES.iloc[7, 8], label=f'{TESTING_SCORES.iloc[7, 0]}: AUC = {TESTING_SCORES.iloc[7, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[8, 7], TESTING_SCORES.iloc[8, 8], label=f'{TESTING_SCORES.iloc[8, 0]}: AUC = {TESTING_SCORES.iloc[8, 6]:.2f}')
plt.plot(TESTING_SCORES.iloc[9, 7], TESTING_SCORES.iloc[9, 8], label=f'{TESTING_SCORES.iloc[9, 0]}: AUC = {TESTING_SCORES.iloc[9, 6]:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


### 外部验证roc
model = GradientBoostingClassifier(**p)
model.fit(X_train_val, y_train_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
# ????ROC????
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

# ????AUC?
from sklearn.metrics import auc 
roc_auc = auc(fpr, tpr)
# ????ROC????
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'GradientBoostingClassifier (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()





