from math import nan

from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

def get_numpy_version():
    print(f"Numpy version: {np.__version__}")
class RandomImputer (OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    
    def __init__(self, nan_value=None, random_state=None):
        if nan_value is None :
            self.nan_value = ['nan']
        elif isinstance(nan_value, str) and nan_value.lower() == 'nan':
            self.nan_value = [nan_value]
        elif isinstance(nan_value, str):
            self.nan_value = [nan_value, 'nan']
        elif isinstance(nan_value, list):
            self.nan_value = nan_value + ['nan']
        elif isinstance(nan_value, list) and 'nan' in nan_value:
            self.nan_value = nan_value
            
        if random_state is None:
            self.random_state = None
        elif isinstance(random_state, int):
            self.random_state = random_state
        else:
            raise ValueError("random_state should be an integer or None.")
    
    def values_counts(self, X):
        
        unique, counts = np.unique(X, return_counts=True)
        
        nan_value_mask = np.isin(unique, self.nan_value)
        
        na_mask = (unique != unique) | (unique == None) | nan_value_mask
        
        indexes = np.argwhere(na_mask).squeeze()
                
        if indexes.size > 0:
            unique_clean = np.delete(unique, indexes)
            counts_clean = np.delete(counts, indexes)
        else:
            unique_clean = unique
            counts_clean = counts
        
        self.unique_.append(unique_clean)
        self.weights_.append(counts_clean / counts_clean.sum())
    
    def fit(self, X, y=None):
        
        self.unique_ = []
        self.weights_ = []
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            shape = X.shape
            if len(shape) == 1: 
                n_cols = 1
            else:
                n_cols = shape[1]
            X_ = np.array(X, copy=True, dtype=np.str_)
        else:
            shape = X.shape
            if len(shape) == 1: 
                n_cols = 1
            else:
                n_cols = shape[1]
            X_ = np.array(X, copy=True, dtype=np.str_)
        
        if n_cols == 1:
            self.values_counts(X_.squeeze())
            return self
        
        for col in range(n_cols):
            self.values_counts(X_[:, col])
        return self
    
    def transform(self, X, y=None):
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            shape = X.shape
            if len(shape) == 1: 
                n_cols = 1
            else:
                n_cols = shape[1]
            X_ = np.array(X, copy=True, dtype=np.str_)
        else:
            shape = X.shape
            if len(shape) == 1: 
                n_cols = 1
            else:
                n_cols = shape[1]
            X_ = np.array(X, copy=True, dtype=np.str_)
        
        rng = np.random.default_rng(self.random_state)
        
        if n_cols == 1:
            X_ = X_.squeeze()
            nan_value_mask = np.isin(X_, self.nan_value)
            
            na_mask = (X_ != X_) | (X_ == None) | nan_value_mask
            
            if na_mask.sum() > 0:
                X_[na_mask] = rng.choice(self.unique_[0], size=na_mask.sum(), p=self.weights_[0])
            return X_
            
        for col in range(n_cols):
            
            nan_value_mask = np.isin(X_[:, col], self.nan_value)
            
            na_mask = (X_[:, col] != X_[:, col]) | (X_[:, col] == None) | nan_value_mask
            
            if na_mask.sum() > 0:
                X_[na_mask, col] = rng.choice(self.unique_[col], size=na_mask.sum(), p=self.weights_[col])
        return X_

def clean_authorities_contacted(X):
    X_ = np.copy(X) # Make a copy to avoid modifying the original data
    
    is_none = (X_[:, 0] == None) # check for none values 
    is_nan = (X_[:, 0] != X_[:, 0]) # check for nan values.
        
    is_null_authorities = is_none | is_nan # Combine both conditions to identify null values in authorities_contacted
    
    mask_police = is_null_authorities & (X_[:, 1] == 'YES')
    mask_unknown = is_null_authorities & (X_[:, 1] != 'YES')
    
    X_[mask_police, 0] = 'Police'
    X_[mask_unknown, 0] = 'Unknown'
    
    return X_

CleanAuthoritiesContacted = FunctionTransformer(clean_authorities_contacted, feature_names_out='one-to-one', validate=False)

# Group high-cardinality categoricals into meaningful buckets
def group_hobby(hobby):
    high_risk = ['bungie-jumping', 'base-jumping', 'skydiving', 'paintball']
    active = ['exercise', 'cross-fit', 'basketball', 'golf', 'hiking', 'polo', 'camping', 'kayaking', 'yoga']
    if hobby in high_risk:
        return 'high_risk'
    elif hobby in active:
        return 'active'
    else:
        return 'leisure'

def group_occupation(occ):
    white_collar = ['exec-managerial', 'prof-specialty', 'tech-support', 'adm-clerical', 'protective-serv', 'sales']
    blue_collar = ['craft-repair', 'machine-op-inspct', 'transport-moving', 'handlers-cleaners', 'farming-fishing', 'priv-house-serv']
    if occ in white_collar:
        return 'white_collar'
    elif occ in blue_collar:
        return 'blue_collar'
    else:
        return 'other'

def group_auto_make(make):
    luxury = ['Mercedes', 'BMW', 'Audi', 'Porsche', 'Lexus', 'Jeep', 'Land Rover', 'Suburu']
    if make in luxury:
        return 'luxury'
    return 'mainstream'

# vectorize the grouping functions for efficient application to FunctionTransformer and direct use on DataFrame columns
v_group_hobby = np.vectorize(group_hobby)
v_group_occupation = np.vectorize(group_occupation)
v_group_auto_make = np.vectorize(group_auto_make)

# Definition of feature transformers for grouping:
GroupHobbyTransformer = FunctionTransformer(v_group_hobby, feature_names_out='one-to-one', validate=False)
GroupOccupationTransformer = FunctionTransformer(v_group_occupation, feature_names_out='one-to-one', validate=False)
GroupAutoMakeTransformer = FunctionTransformer(v_group_auto_make, feature_names_out='one-to-one', validate=False)