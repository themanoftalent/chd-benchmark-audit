"""Reproduction workflow for the historical coronary disease benchmark audit."""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalch','exang','oldpeak','slope']
NUMERIC = ['age','trestbps','chol','thalch','oldpeak']
KEY = FEATURES + ['target']
UCI_COLUMNS = FEATURES + ['ca','thal','num']
SOURCES = {'Hungarian':'processed.hungarian.data', 'Switzerland':'processed.switzerland.data',
           'Long Beach VA':'processed.va.data', 'Cleveland':'processed.cleveland.data'}


def load_data(folder, coding):
    frames = []
    for source, filename in SOURCES.items():
        d = pd.read_csv(folder / filename, header=None, names=UCI_COLUMNS, na_values=['?'], skipinitialspace=True).apply(pd.to_numeric, errors='raise')
        if not d.num.isin([0,1,2,3,4]).all():
            raise ValueError(f'Invalid outcome in {filename}')
        d['target'] = (d.num > 0).astype(int)
        d['source'] = source
        d['record_id'] = source + ':' + d.index.astype(str)
        frames.append(d[KEY + ['source','record_id']])
    raw = pd.concat(frames, ignore_index=True)
    comp = pd.read_csv(folder / 'heart_disease_uci.csv', na_values=['?'])
    aliases = {'resting bp s':'trestbps','chest pain type':'cp','cholesterol':'chol',
               'fasting blood sugar':'fbs','resting ecg':'restecg','max heart rate':'thalch',
               'exercise angina':'exang','ST slope':'slope','st slope':'slope','heart disease':'target',
               'heartdisease':'target','heart disease':'target','num':'target','output':'target','trtbps':'trestbps','thalach':'thalch',
               'exng':'exang','slp':'slope'}
    comp.columns = comp.columns.str.strip().str.lower()
    comp = comp.rename(columns=aliases)
    missing=set(KEY)-set(comp.columns)
    if missing: raise ValueError(f'Missing compilation columns: {sorted(missing)}')
    if comp.columns.duplicated().any(): raise ValueError('Duplicate compilation column names after harmonisation')
    comp = comp[KEY].apply(pd.to_numeric, errors='raise')
    if not comp.target.isin([0,1,2,3,4]).all():
        raise ValueError('Invalid compilation outcome')
    comp['target'] = (comp.target > 0).astype(int)
    if coding == 'zero-based':
        comp['cp'] += 1
        comp['slope'] += 1
    for d in [raw, comp]:
        for col, allowed in {'sex':[0,1],'cp':[1,2,3,4],'fbs':[0,1],
                             'restecg':[0,1,2],'exang':[0,1],'slope':([0,1,2,3] if d is comp else [1,2,3])}.items():
            if not d[col].dropna().isin(allowed).all():
                raise ValueError(f'Unexpected {col} codes; verify source encoding')
    comp['record_id'] = 'comp:' + comp.index.astype(str)
    stat = pd.read_csv(folder / 'heart.dat', sep=r'\s+', header=None, names=UCI_COLUMNS).apply(pd.to_numeric, errors='raise')
    if not stat.num.isin([1,2]).all():
        raise ValueError('Unexpected Statlog endpoint')
    stat['target'] = (stat.num == 2).astype(int)
    stat['source']='Statlog'
    stat['record_id']='Statlog:'+stat.index.astype(str)
    return raw, comp, stat[KEY+['source','record_id']]


def preprocessor(features):
    num = [c for c in features if c in NUMERIC]
    cat = [c for c in features if c not in num]
    return ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                         ('scale',StandardScaler())]),num),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
                         ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])


def pipe(features, model):
    return Pipeline([('prep',preprocessor(features)),('clf',clone(model))])


def metrics(y,p):
    return {'auroc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
            'accuracy':accuracy_score(y,p>=.5),'brier':brier_score_loss(y,p)}



from sklearn.model_selection import ParameterGrid

def models(seed=2026):
    return {
        'LR':(LogisticRegression(max_iter=5000),{'C':[.01,.03,.1,.3,1,3,10,30]}),
        'kNN':(KNeighborsClassifier(),{'n_neighbors':[5,11,19,31],'weights':['uniform','distance']}),
        'SVM':(SVC(probability=True,random_state=seed),{'C':[.03,.1,.3,1,3,10,30,100]}),
        'DT':(DecisionTreeClassifier(random_state=seed),{'max_depth':[2,4,6,10],'min_samples_leaf':[5,15]}),
        'RF':(RandomForestClassifier(n_estimators=150,random_state=seed,n_jobs=1),{'max_depth':[3,6,10,None],'min_samples_leaf':[3,10]}),
        'XGB':(XGBClassifier(n_estimators=150,eval_metric='logloss',random_state=seed,n_jobs=1),{'max_depth':[2,3,4,5],'learning_rate':[.03,.10]})}

def clean(d, mode='median'):
    d=d.copy()
    d.loc[d.trestbps==0,'trestbps']=np.nan
    if mode!='zero': d.loc[d.chol==0,'chol']=np.nan
    if mode=='indicator': d['chol_missing']=d.chol.isna().astype(int)
    return d

def split(d,n,seed):
    return list(StratifiedKFold(n,shuffle=True,random_state=seed).split(d,d.target))

def overlap_count(train,test):
    a=pd.util.hash_pandas_object(train[FEATURES].astype(float),index=False)
    b=pd.util.hash_pandas_object(test[FEATURES].astype(float),index=False)
    return int(b.isin(a).sum())

def tune(d,features,name,seed,jobs=1,folds=3):
    m,g=models()[name]
    s=GridSearchCV(pipe(features,m),{'clf__'+k:v for k,v in g.items()},scoring='roc_auc',cv=split(d,folds,seed),n_jobs=jobs,error_score='raise')
    s.fit(d[features],d.target)
    return s.best_estimator_,s.best_params_

def ci(y,p,seed=2026,n=2000):
    y=np.asarray(y);p=np.asarray(p); rng=np.random.default_rng(seed)
    ix=[np.flatnonzero(y==k) for k in [0,1]]
    if any(len(i)==0 for i in ix): return [None,None]
    from scipy.stats import rankdata
    vals=[]; n0,n1=map(len,ix)
    for _ in range(n):
        a=np.concatenate([rng.choice(i,len(i),replace=True) for i in ix]); r=rankdata(p[a],method='average'); vals.append((r[n0:].sum()-n1*(n1+1)/2)/(n0*n1))
    return np.quantile(vals,[.025,.975]).tolist()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v),allow_nan=False))
