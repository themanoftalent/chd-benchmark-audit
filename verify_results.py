"""Compare regenerated outputs with the numerical reference in R4 (not model inputs)."""
import argparse,json,math,ast
from pathlib import Path
import pandas as pd

def verify(root, tol=None):
    checks=[]
    def eq(label,actual,expected,digits=None):
        if digits is None:
            ok = (actual==expected)
        elif tol is not None:
            ok = math.isfinite(float(actual)) and abs(float(actual)-expected) <= tol
        else:
            ok = math.isfinite(float(actual)) and abs(float(actual)-expected) <= 0.5*10**(-digits)+1e-10
        checks.append(dict(check=label,actual=actual,reference=expected,pass_check=bool(ok)))
    def read(name): return json.loads((root/name).read_text())
    def csv(name):return pd.read_csv(root/name)
    a=read('audit.json');eq('original rows',a['original']['rows'],920);eq('original repeats',a['original']['duplicates'],2);eq('compilation rows',a['compilation']['rows'],1190);eq('compilation repeats',a['compilation']['duplicates'],272)
    l=csv('anchor_sensitivity.csv').set_index('anchors');
    for k,values in {3:[1190,1186,4,0,5],4:[1135,1135,0,55,0],5:[1135,1135,0,55,0],6:[1131,1131,0,59,0]}.items():
        for c,v in zip(['candidates','one_family','multiple_families','unmatched','discordant'],values):eq(f'anchor {k} {c}',int(l.loc[k,c]),v)
    f=csv('fills.csv');
    for field,y,code,count in [('slope',0,1,174),('slope',1,2,80),('slope',1,0,1),('fbs',0,0,14),('fbs',1,1,76)]:eq(f'{field} label={y} value={code}',int(((f.field==field)&(f.target==y)&(f.value==code)).sum()),count)
    p=read('primary.json');
    for name,values in {'LR':[.872,.024,.794,.873],'kNN':[.877,.015,.807,.875],'SVM':[.868,.024,.801,.869],'DT':[.809,.035,.734,.810],'RF':[.874,.027,.798,.873],'XGB':[.872,.030,.789,.872]}.items():
        for c,v in zip(['mean_auroc','sd_auroc','accuracy','pooled_auroc'],values):eq(f'{name} {c}',p[name][c],v,3)
    q=csv('ablation.csv').set_index(['model','subset']);
    for name,vals in {'LR':[.872,.838,.734,.723],'RF':[.874,.849,.743,.725],'XGB':[.872,.847,.742,.723]}.items():
        for k,v in zip(['all11','no_exercise','risk5','risk4'],vals):eq(f'ablation {name} {k}',q.loc[(name,k),'mean_auroc'],v,3)
    q=csv('cholesterol.csv').set_index(['dataset','treatment']);
    for name,vals in {'original':[.884,.872,.877,.868],'compilation':[.931,.928,.931,.929]}.items():
        for k,v in zip(['zero','median','indicator','omitted'],vals):eq(f'cholesterol {name} {k}',q.loc[(name,k),'mean_auroc'],v,3)
    q=csv('sources.csv').set_index('source');
    for name,vals in {'Hungarian':[293,106,.872,.880],'Switzerland':[123,115,.771,.715],'Long Beach VA':[199,148,.713,.575],'Cleveland':[303,139,.858,.855]}.items():
        for k,v in zip(['n','positive','heldout','within'],vals):eq(f'source {name} {k}',q.loc[name,k],v,None if k in ['n','positive'] else 3)
    eq('source probe',read('source_probe.json')['balanced_accuracy'],.638,3)
    q=csv('subgroups.csv').set_index('group');
    for name,vals in {'women':[193,50,.854,.793,.909],'men':[725,458,.854,.825,.881],'chol_unavailable':[201,163,.800,.711,.879],'chol_recorded':[717,345,.868,.843,.893]}.items():
        for k,v in zip(['n','positive','auroc'],vals[:3]):eq(f'{name} {k}',q.loc[name,k],v,None if k!='auroc' else 3)
        for k,a,b in zip(['lower','upper'],ast.literal_eval(q.loc[name,'ci']),vals[3:]):eq(f'{name} {k}',a,b,3)
    q=read('calibration.json');
    for name,vals in {'LR':[.142,1.085,-.007],'RF':[.143,1.284,-.015],'XGB':[.144,1.211,-.015],'XGB_recalibrated':[.145,1.107,-.022]}.items():
        for k,v in zip(['brier','slope','intercept'],vals):eq(f'calibration {name} {k}',q[name][k],v,3)
    q=csv('thresholds.csv').set_index('rule');
    for name,vals in {'fixed':[421,107,303,87,.500,.500,.500],'youden':[435,124,286,73,.470,.396,.522],'sensitivity90':[463,143,267,45,.412,.364,.430],'sensitivity95':[488,205,205,20,.264,.233,.306]}.items():
        for k,v in zip(['tp','fp','tn','fn','median','minimum','maximum'],vals):eq(f'threshold {name} {k}',q.loc[name,k],v,None if k in ['tp','fp','tn','fn'] else 3)
    q=csv('duplicates.csv');eq('matched seed count',len(q),20);eq('duplicate AUROC mean',q.delta_auroc.mean(),.030,3);eq('duplicate accuracy mean',q.delta_accuracy.mean(),.043,3);eq('duplicate AUROC SD',q.delta_auroc.std(),.011,3);eq('duplicate accuracy SD',q.delta_accuracy.std(),.017,3)
    e=read('esc.json');eq('ESC n',e['n'],418);eq('ESC positive',e['positive'],116);eq('ESC AUROC',e['esc_auroc'],.687,3);eq('ESC LR AUROC',e['lr_auroc'],.774,3)
    q=csv('A8_folds.csv');
    for i,vals in enumerate([[57,23,38,66],[60,19,44,61],[57,24,38,65],[58,30,37,58],[61,27,42,53]]):
        for k,v in zip(['Hungarian','Switzerland','Long Beach VA','Cleveland'],vals):eq(f'fold {i+1} {k}',q.loc[i,k],v)
    m=read('metrics.json');
    for fold,count in m['predictor_overlap_by_fold'].items():eq(f'primary fold {fold} predictor overlap',count,0)
    near=csv('near_duplicates.csv');
    for i,v in enumerate([0,0,0,0,1]):eq(f'near-duplicate rule {i+1}',near.loc[i,'pairs'],v)
    if (root/'A10_explanations.json').exists():
        e=read('A10_explanations.json');
        for k,v in {'global_spearman':.887,'global_overlap':.767,'interventional_spearman':.991,'local_spearman':.826,'local_overlap':.643}.items():eq(k,e[k],v,3)
        s=csv('shap_comparison.csv');eq('path top 3',s.nlargest(3,'path').feature.tolist(),['cp','sex','exang']);eq('interventional top 3',s.nlargest(3,'interventional').feature.tolist(),['cp','sex','exang'])
    else:checks.append(dict(check='SHAP/LIME outputs',actual='missing',reference='required',pass_check=False))
    return checks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--tol',type=float,default=None,help='Optional absolute tolerance threshold override for floating-point comparisons (e.g. 0.02)')
    a=p.parse_args()
    try:checks=verify(a.results, tol=a.tol)
    except (FileNotFoundError,KeyError,ValueError) as exc:raise SystemExit(f'Incomplete or incompatible outputs: {exc}')
    d=pd.DataFrame(checks);d.to_csv(a.results/'verification.csv',index=False)
    fails=d[~d.pass_check]
    if len(fails)>0:
        print(fails.to_string(index=False))
    print(f'{d.pass_check.sum()}/{len(d)} checks match R4 reference{" with tolerance " + str(a.tol) if a.tol is not None else " precision."}')
    raise SystemExit(0 if d.pass_check.all() else 1)
if __name__=='__main__':main()
