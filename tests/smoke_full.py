"""Nonclinical integration check. Reduced search grids are used only here."""
import sys,tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
tests_dir = Path(__file__).resolve().parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from unittest.mock import patch
import pandas as pd
import numpy as np
import common,analysis,generate_figures,verify_results
from test_pipeline import fixture

def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); data=root/'data'; data.mkdir(); out=root/'results'; figures=root/'figures'
        raw=fixture(920); raw.loc[:39,'chol']=0
        raw.loc[293,common.KEY]=raw.loc[0,common.KEY].to_numpy()
        raw.loc[616,common.KEY]=raw.loc[417,common.KEY].to_numpy()
        raw['ca']=0;raw['thal']=3;raw['num']=raw.target
        offsets=[0,294,417,617,920]
        for i,(source,name) in enumerate(common.SOURCES.items()):
            raw.iloc[offsets[i]:offsets[i+1]][common.UCI_COLUMNS].to_csv(data/name,index=False,header=False)
        stat=raw.iloc[617:887].copy();stat['num']=stat.target+1
        stat[common.UCI_COLUMNS].to_csv(data/'heart.dat',sep=' ',index=False,header=False)
        pd.concat([raw[common.KEY],stat[common.KEY]],ignore_index=True).to_csv(data/'heart_disease_uci.csv',index=False)
        cfg=common.models(); small={}
        for name,(m,grid) in cfg.items():
            if name in ['RF','XGB']:m.set_params(n_estimators=8)
            small[name]=(m,{k:[v[0]] for k,v in grid.items()})
        with patch('common.models',return_value=small),patch('analysis.models',return_value=small),patch.object(sys,'argv',['run.py','--data',str(data),'--output',str(out),'--skip-explanations']):
            analysis.main()
        with patch.object(sys,'argv',['generate_figures.py','--results',str(out),'--output',str(figures)]):generate_figures.main()
        checks=verify_results.verify(out)
        assert any(not q['pass_check'] for q in checks), 'Synthetic inputs must not pass the clinical numerical verifier'
        assert len(list(figures.glob('*.png')))==7
        assert len(pd.read_csv(out/'threshold_folds.csv'))==20
        assert len(pd.read_csv(out/'duplicates.csv'))==20
        print('PASS: synthetic integration, all non-explanation analysis stages, seven figures, mismatch detection.')
if __name__=='__main__':main()
