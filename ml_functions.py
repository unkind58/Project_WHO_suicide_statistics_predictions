import math
import random
import pathlib
import missingno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


def refactor_titles(df: pd.DataFrame, sub_chars=[' ','-',':'], drop_chars=['(',')','[',']']) -> list:
    '''
    Function generalizes all column titles, i.e. lowers all cases, makes characters 
    substitution from one input list and drops characters from another input list.
    Prints how many substitutions were made.
    Parameters:
    -----------
    (1) df --> DataFrame which columns function should transform;
    (2) sub_chars --> characters to substitute with '_';
    (3) drop_chars --> characters to drop;
    -----------
    Returns list with new column titles.
    '''
    list_columns = [item.lower() for item in (df.columns.tolist())]   
    for sub_char in sub_chars:
        counter_sub = 0
        for i in range(len(list_columns)):
            if sub_char in list_columns[i]:
                counter_sub += 1
                list_columns[i] = list_columns[i].replace(sub_char, '_')
            else:
                continue
        print(f'Substituotion of "{sub_char}" occured {counter_sub} times.')
    for drop_char in drop_chars:
        counter_drop = 0
        for i in range(len(list_columns)):
            if drop_char in list_columns[i]:
                list_columns[i] = list_columns[i].replace(drop_char, '') 
            else:
                continue
            print(f'Drop of "{drop_char}" occured {counter_drop} times.')              
    return list_columns


def nan_columns_by_country(df: pd.DataFrame, column_mark: str) -> dict:
    '''Function identifies column names in a given DataFrame containing at 
    least one NaN value, puts them into a list and assigns it to variable
    'columns_nan_list'. Then it takes input 'column_nan' column and 
    checks if it is in 'columns_nan_list'. If not, proceed further. 
    The last action is iterations through each NaN column and grouping
    by 'column_nan' categories in it.
    Parameters:
    -----------
    (1) df --> given DataFrame;
    (2) column_nan --> categorical column, each one will be filtered for 
    NaN values;
    -----------
    Returns dictionary with given column categorical values and count of 
    NaN values in it.
    '''
    columns_nan_list = df.columns[df.isna().any()].tolist()
    if column_mark in columns_nan_list:
        return print(f'Column {column_nan} has NaN values, please choose another column')
    set_of_nan = {}
    for column in columns_nan_list:
        set_of_nan[column] = list(set(df[df[column].isnull()][column_mark]))
    for key, value in sorted(set_of_nan.items()):
        print(f'* {len([item for item in value if item])} categories in column "{column_mark}" with at least one NaN in "{key}".')
    return set_of_nan


def backfilling_nan_by_country(df: pd.DataFrame, columns_nan_dict: dict, target_column: str, category_column: str):
    '''Function fills NaN values directly into given DataFrame using 
    'backfill' method.
    Parameters:
    -----------
    (1) df --> given DataFrame;
    (2) columns_nan_dict --> dictionary with all NaN columns as a key and 
    all NaN categories from category_column;
    (3) target_column --> column name where NaN values will be filled using 
    'backfill' method;
    (4) category_column --> column name where categories have NaN values 
    in 'target_column';
    -----------
    Returns text confirmation that NaN values were filled, otherwise text 
    confirmation that no NaN occured.
    '''
    if df[target_column].isnull().any():
        for nan_category in columns_nan_dict[target_column]:
            mask = (df[category_column] == nan_category)
            df.loc[mask,target_column] = df.loc[mask,target_column].fillna(method='backfill')
        return f'NaN values were filled in "{target_column}"" column'
    else:
        return f'There are no NaN values "{target_column}"" column'

    
def missing_values(df: pd.DataFrame):
    ''' Functions uses missingno library and prints each DataFrame's 
    column name and count of NaN which were found in it.
    Parameters:
    -----------
    (1) df --> given DataFrame;
    -----------
    Returns a chart bar with each feature non-NaN value count.
    '''
    for column in df.columns[df.isnull().any(axis=0)]:
        print(f'Column "{column}" has {df[column].isnull().sum()} missing values.')
    return missingno.bar(df)


def rmse(x: np.ndarray,y: np.ndarray): 
    '''Root Mean Square Error (RMSE) is the standard deviation of the residuals 
    (prediction errors). Function takes predictions based on trained features as 
    'x' and answers as 'y'.
    Parameters:
    -----------
    (1) x --> predictions ;
    (2) y --> observed values (known results);
    -----------
    Returns scoring values in the following order: 
    [training rmse, validation rmse, r² for training set, r² for validation set, 
    oob_score_]
    '''
    return math.sqrt(((x-y)**2).mean())


def print_score(m, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid:pd.Series):
    '''Function takes a model and calculates and prints its RMSE values and r² 
    scores for train and validation set. Also attaches oob_score for Random 
    Forest model.
    Parameters:
    -----------
    (1) m --> given model;
    (2) X_train --> training set of independent features;
    (3) X_valid --> validation set of independent feature;
    (4) y_train --> training set of dependent features;
    (5) y_valid --> validation set of dependent feature;
    -----------
    Returns scoring values in the following order: 
    [training rmse, validation rmse, r² for training set, r² for validation set, 
    mae for validation set, oob_score_]
    '''
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid),
           mean_absolute_error(y_train, m.predict(X_train)),
           mean_absolute_error(y_valid, m.predict(X_valid))]
    CRED = '\033[91m'
    CEND = '\033[0m'
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    if hasattr(m, 'oob_score_'): print(CRED + f'     oob Score: \t\t{round(res[6],8)}'+ CEND)
    return print(f' \
    RMSE Train set: \t\t{round(res[0],8)}  \n \
    RMSE Validation set: \t{round(res[1],8)} \n \
    r² Train set: \t\t{round(res[2],8)} \n \
    r² Validation set: \t{round(res[3],8)} \n \
    MAE Train set: \t\t{round(res[4],8)} \n \
    MAE Validation set: \t{round(res[5],8)}')


def get_mae(max_leaf_nodes: list, train_X: pd.DataFrame, val_X: pd.DataFrame,train_y: pd.Series, val_y:pd.Series) -> np.float64:
    '''Mean Absolute Error (MAE) evaluates the average dimension
    of the error in a predictions set. Function takes X and y 
    from the training and validation sets.
    Parameters:
    -----------
    (1) max_leaf_nodes --> list of possible 'max_leaf_nodes' values;
    (2) train_X --> training set of independent features;
    (3) val_X --> validation set of independent feature;
    (4) train_y --> training set of dependent features;
    (5) val_y --> validation set of dependent feature;
    -----------
    Returns MAE value.
    '''
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=158)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def print_score_scaler(m,X_train_scaler: pd.DataFrame, X_valid_scaler: pd.DataFrame, y_train: pd.Series, y_valid:pd.Series):
    '''Function takes a model and calculates and prints its RMSE values and r² 
    scores for train and validation sets, but both X sets should be scaled.
    Parameters:
    -----------
    (1) m --> given model;
    (2) X_train --> training set of independent features;
    (3) X_valid --> validation set of independent feature;
    (4) y_train --> training set of dependent features;
    (5) y_valid --> validation set of dependent feature;
    -----------
    Returns scoring values in the following order: 
    [training rmse, validation rmse, r² for training set, r² for validation set, 
    mae for validation set]
    '''
    res = [rmse(m.predict(X_train_scaler), y_train),
           rmse(m.predict(X_valid_scaler), y_valid),
           m.score(X_train_scaler, y_train), m.score(X_valid_scaler, y_valid),
           mean_absolute_error(y_train, m.predict(X_train_scaler)),
           mean_absolute_error(y_valid, m.predict(X_valid_scaler))]
    return print(f' \
    RMSE Train set: \t\t{round(res[0],8)}  \n \
    RMSE Validation set: \t{round(res[1],8)} \n \
    r² Train set: \t\t{round(res[2],8)} \n \
    r² Validation set: \t{round(res[3],8)} \n \
    MAE Train set: \t\t{round(res[4],8)} \n \
    MAE Validation set: \t{round(res[5],8)}')

def print_score_log(m, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid:pd.Series):
    '''Function takes a model and calculates and prints its RMSE values and r² 
    scores for train and validation sets, but both y sets should be logarithmed.
    Parameters:
    -----------
    (1) m --> given model;
    (2) X_train --> training set of independent features;
    (3) X_valid --> validation set of independent feature;
    (4) y_train --> training set of dependent features;
    (5) y_valid --> validation set of dependent feature;
    -----------
    Returns scoring values in the following order: 
    [training rmse, validation rmse, r² for training set, r² for validation set, 
    mae for validation set]
    '''
    res = [rmse(np.expm1(m.predict(X_train)), np.expm1(y_train)),
           rmse(np.expm1(m.predict(X_valid)), np.expm1(y_valid)),
           r2_score(np.expm1(y_train), np.expm1(m.predict(X_train))), 
           r2_score(np.expm1(y_valid), np.expm1(m.predict(X_valid))),
           mean_absolute_error(np.expm1(y_train), np.expm1(m.predict(X_train))),
           mean_absolute_error(np.expm1(y_valid), np.expm1(m.predict(X_valid)))]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    if hasattr(m, 'oob_score_'): print(f'     oob Score: \t\t{round(res[6],8)}')
    return print(f' \
    RMSE Train set: \t\t{round(res[0],8)}  \n \
    RMSE Validation set: \t{round(res[1],8)} \n \
    r² Train set: \t\t{round(res[2],8)} \n \
    r² Validation set: \t{round(res[3],8)} \n \
    MAE Train set: \t\t{round(res[4],8)} \n \
    MAE Validation set: \t{round(res[5],8)}')
