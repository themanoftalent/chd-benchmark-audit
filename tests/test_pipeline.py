import unittest, tempfile, zipfile, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from common import *
from analysis import nested, summary, duplicate_test, esc_prob, overlap
from linkage import audit
from recalibrate import calibration, fit_cal, logits
from fetch_inputs import unpack
from generate_figures import workflow


def fixture(n=160, seed=18):
    rng=np.random.default_rng(seed)
    y=np.tile([0,1],n//2)
    d=pd.DataFrame(dict(age=rng.integers(35,78,n),sex=rng.integers(0,2,n),cp=rng.integers(1,5,n),trestbps=rng.integers(95,170,n),chol=rng.integers(140,300,n),fbs=rng.integers(0,2,n),restecg=rng.integers(0,3,n),thalch=rng.integers(85,185,n),exang=rng.integers(0,2,n),oldpeak=rng.uniform(0,4,n),slope=rng.integers(1,4,n),target=y))
    d.loc[y==1,'age']+=5
    d['source']='Cleveland';d['record_id']='test:'+d.index.astype(str)
    return d

class PipelineTests(unittest.TestCase):
    def test_search_grids(self):
        self.assertEqual(len(models()),6)
        self.assertTrue(all(len(list(ParameterGrid(g)))==8 for m,g in models().values()))

    def test_cleaning(self):
        d=fixture(20);d.loc[0,['chol','trestbps']]=0
        a=clean(d,'indicator');self.assertTrue(pd.isna(a.loc[0,'chol']));self.assertTrue(pd.isna(a.loc[0,'trestbps']));self.assertEqual(a.loc[0,'chol_missing'],1);self.assertEqual(d.loc[0,'chol'],0)
        self.assertEqual(clean(d,'zero').loc[0,'chol'],0)

    def test_all_classifiers(self):
        d=clean(fixture());a,b=d.iloc[:120],d.iloc[120:]
        for name,(m,g) in models().items():
            with self.subTest(model=name):
                est=pipe(FEATURES,m).fit(a[FEATURES],a.target);p=est.predict_proba(b[FEATURES]);self.assertEqual(p.shape,(40,2));self.assertTrue(np.all(np.isfinite(p)));np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)

    def test_nested_lr(self):
        d=clean(fixture());p,f,r=nested(d,FEATURES,'LR');self.assertEqual(len(p),len(d));self.assertEqual(p.row.nunique(),len(d));self.assertEqual(len(f),5);self.assertEqual(len(r),0);self.assertTrue(p.groupby('fold').predictor_overlap_count.first().eq(0).all());self.assertTrue(0<=summary(p)['pooled_auroc']<=1)

    def test_threshold_crossfit_and_calibration(self):
        d=clean(fixture());m=XGBClassifier(n_estimators=8,max_depth=2,random_state=2026,n_jobs=1)
        # Reduce search cost for this software test only; production grids are unchanged.
        with patch('common.models',return_value={'XGB':(m,{'max_depth':[2]})}):
            p,f,r=nested(d,FEATURES,'XGB',outer=2,operating=True)
        self.assertEqual(len(r),8);self.assertEqual(set(r.rule),{'fixed','youden','sensitivity90','sensitivity95'});self.assertTrue((r.groupby('rule')[['tp','fp','tn','fn']].sum().sum(axis=1)==len(d)).all());self.assertTrue(p.recalibrated.between(0,1).all());self.assertTrue(np.isfinite(calibration(p.target,p.probability)['slope']))

    def test_bootstrap_auc(self):
        self.assertEqual(ci([0,0,1,1],[.1,.2,.8,.9],n=50),[1.,1.]);self.assertEqual(ci([0,0,1,1],[.5]*4,n=50),[.5,.5])

    def test_esc_lookup(self):
        self.assertEqual(esc_prob(pd.Series(dict(age=35,sex=1,cp=1))),.03);self.assertEqual(esc_prob(pd.Series(dict(age=75,sex=0,cp=3))),.10);self.assertTrue(np.isnan(esc_prob(pd.Series(dict(age=40,sex=0,cp=4)))))

    def test_loading_and_linkage(self):
        d=fixture(12);d['ca']=0;d['thal']=3;d['num']=d.target
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for k,file in SOURCES.items():d[UCI_COLUMNS].iloc[:2].to_csv(p/file,header=False,index=False)
            stat=d[UCI_COLUMNS].iloc[:2].copy();stat['num']+=1;stat.to_csv(p/'heart.dat',sep=' ',header=False,index=False)
            c=d[KEY].iloc[:2].copy();c.loc[c.index[0],'slope']=0;c.to_csv(p/'heart_disease_uci.csv',index=False)
            a,b,s=load_data(p,'one-based');self.assertEqual(len(a),8);self.assertIn(0,b.slope.to_list());self.assertEqual(s.target.to_list(),[0,1])
            audit(a,b,s,p);self.assertTrue((p/'anchor_sensitivity.csv').exists());self.assertTrue((p/'near_duplicates.csv').exists())

    def test_source_linkage_missing_and_label_exclusion(self):
        a=fixture(4);a.loc[0,'slope']=np.nan;b=a.iloc[[0]].copy();b.loc[0,'slope']=2;b.loc[0,'target']=1-a.loc[0,'target'];s=a.iloc[:0].copy()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);audit(a,b,s,p);l=pd.read_csv(p/'row_linkage.csv');self.assertEqual(l.candidate_count.iloc[0],1);self.assertTrue(l.discordant.iloc[0]);f=pd.read_csv(p/'fills.csv');self.assertIn('slope',f.field.to_list())

    def test_duplicate_key_and_conflicts(self):
        d=fixture(20);x=pd.concat([d,d.iloc[[0]],d.iloc[[1]].assign(target=0)],ignore_index=True)
        with tempfile.TemporaryDirectory() as t:
            a=audit(x,x,d.iloc[:0],Path(t));self.assertEqual(a['original']['duplicates'],1);self.assertEqual(a['original']['predictor_duplicates'],2);self.assertEqual(a['original']['conflicting_vectors'],1)

    def test_archive_extraction(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);z=p/'a.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('nested/heart.dat','test');f.writestr('../unsafe','ignore')
            unpack(z,p,['heart.dat']);self.assertEqual((p/'heart.dat').read_text(),'test');self.assertFalse((p.parent/'unsafe').exists())

    def test_shap_tree(self):
        import shap
        d=clean(fixture(80));m=pipe(FEATURES,XGBClassifier(n_estimators=8,max_depth=2,n_jobs=1)).fit(d[FEATURES],d.target);a=m.named_steps['prep'].transform(d[FEATURES]);v=shap.TreeExplainer(m.named_steps['clf'],feature_perturbation='tree_path_dependent').shap_values(a[:4]);self.assertEqual(v.shape,a[:4].shape)

    def test_lime_raw_pipeline(self):
        try:from lime.lime_tabular import LimeTabularExplainer
        except ImportError:self.skipTest('LIME is not installed in this execution environment')
        d=clean(fixture(80));m=pipe(FEATURES,models()['LR'][0]).fit(d[FEATURES],d.target);sentinel=-999.;x=d[FEATURES].fillna(sentinel).to_numpy(float);cat=[i for i,c in enumerate(FEATURES) if c not in NUMERIC];e=LimeTabularExplainer(x,categorical_features=cat,mode='classification',random_state=902)
        def pred(v):return m.predict_proba(pd.DataFrame(v,columns=FEATURES).replace(sentinel,np.nan))
        z=e.explain_instance(x[0],pred,labels=(1,),num_features=11,num_samples=50);self.assertIn(1,z.local_exp)

    def test_workflow_figure(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);workflow(p);self.assertTrue((p/'Figure_1.png').stat().st_size>1000);self.assertTrue((p/'Figure_1.svg').stat().st_size>1000)

if __name__=='__main__':unittest.main()
