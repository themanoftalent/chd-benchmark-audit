import numpy as np
from scipy.special import logit,expit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

def logits(p): return logit(np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8)).reshape(-1,1)
def fit_cal(y,p): return LogisticRegression(C=np.inf,max_iter=5000).fit(logits(p),y)
def calibration(y,p):
    z=logits(p).ravel(); y=np.asarray(y); m=fit_cal(y,p)
    a=minimize_scalar(lambda a:np.sum(np.logaddexp(0,z+a)-y*(z+a))).x
    return dict(brier=float(brier_score_loss(y,p)),slope=float(m.coef_[0,0]),intercept=float(a))
