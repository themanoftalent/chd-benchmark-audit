"""Generate descriptive tables and audit trails from the local clinical inputs."""
from common import *

def export_tables(raw,comp,stat,d,xp,rules,out):
    c=clean(comp.drop_duplicates(KEY).reset_index(drop=True)); refs=pd.concat([raw,stat],ignore_index=True)
    desc=[]; cats=[]
    for tag,frame in [('original',d),('compilation',c)]:
        for y,g in frame.groupby('target'):
            for f in NUMERIC:
                v=g[f].dropna();desc.append(dict(dataset=tag,target=y,feature=f,n=len(v),median=v.median(),q1=v.quantile(.25),q3=v.quantile(.75)))
            for f in [c for c in FEATURES if c not in NUMERIC]:
                for code,n in g[f].value_counts().items(): cats.append(dict(dataset=tag,target=y,feature=f,code=code,n=n))
    pd.DataFrame(desc).to_csv(out/'A2_continuous.csv',index=False);pd.DataFrame(cats).to_csv(out/'A3_categorical.csv',index=False)
    rows=[];exercise=[]
    ex=['thalch','exang','oldpeak','slope']
    for s,g in d.groupby('source'):
        rows.append(dict(source=s,retained=len(g),positive=g.target.sum(),female=(g.sex==0).sum(),complete=g[FEATURES].notna().all(axis=1).sum(),median_age=g.age.median()))
        for y,h in g.groupby('target'):
            m=h[ex].isna();exercise.append(dict(source=s,target=y,rows=len(h),all_four_available=(~m.any(axis=1)).sum(),at_least_one_missing=m.any(axis=1).sum(),all_four_missing=m.all(axis=1).sum()))
    pd.DataFrame(rows).to_csv(out/'A4_cohorts.csv',index=False);pd.DataFrame(exercise).to_csv(out/'A14_exercise.csv',index=False)
    missing=d.groupby('source')[FEATURES].agg(lambda x:x.isna().sum());missing.to_csv(out/'A5_missingness.csv')
    links=pd.read_csv(out/'row_linkage.csv'); offsets=[0,294,417,617,920,1190];names=['Hungarian','Switzerland','Long Beach VA','Cleveland','Statlog']
    if len(comp)==1190:
        # Block labels are descriptive only and never enter the linkage rule.
        links['block']=pd.cut(links.row,offsets,right=False,labels=names); rows=[]
        for s,g in links.groupby('block',observed=True):
            n=g.candidate_count; rows.append(dict(block=s,rows=len(g),any_strict=(n>0).sum(),one_strict=(n==1).sum(),multiple_strict=(n>1).sum(),no_strict=(n==0).sum()))
        pd.DataFrame(rows).to_csv(out/'A6_linkage_multiplicity.csv',index=False)
        originals=refs.groupby('source')[['slope','oldpeak','fbs','chol']].agg(lambda x:x.isna().sum())
        pd.DataFrame(rows).set_index('block').join(originals).to_csv(out/'table2_linkage.csv')
    fills=pd.read_csv(out/'fills.csv'); fills.groupby(['family','field','target']).value.agg(['size','min','max']).to_csv(out/'A7_fills_summary.csv')
    fold=[]
    for f,g in xp.groupby('fold'):
        b=d.iloc[g.row.to_numpy()];fold.append(dict(fold=f+1,train=len(d)-len(b),train_positive=d.target.sum()-b.target.sum(),test=len(b),test_positive=b.target.sum(),**{s:int((b.source==s).sum()) for s in SOURCES}))
    pd.DataFrame(fold).to_csv(out/'A8_folds.csv',index=False)
    rules.groupby('rule')[['tp','fp','tn','fn']].sum().assign(predicted_positive=lambda x:x.tp+x.fp).to_csv(out/'A15_confusion_counts.csv')
    if (out/'shap_stability.csv').exists():
        sh=pd.read_csv(out/'shap_stability.csv');li=pd.read_csv(out/'lime_agreement.csv');ic=pd.read_csv(out/'shap_comparison.csv')
        write_json(out/'A10_explanations.json',{'global_spearman':sh.spearman.mean(),'global_overlap':sh.overlap.mean(),'interventional_spearman':spearmanr(ic.path,ic.interventional).statistic,'local_spearman':li.spearman.mean(),'local_overlap':li.overlap.mean(),'chance_overlap':3/11,'lime_surrogate_r2_mean':li.fidelity.mean()})
