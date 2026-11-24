from common import *
from linkage import audit
from recalibrate import fit_cal,logits,calibration
from sklearn.model_selection import train_test_split
from importlib.metadata import version
from itertools import combinations

SETS={'all11':FEATURES,'no_exercise':[c for c in FEATURES if c not in ['thalch','exang','oldpeak','slope']], 'risk5':['age','sex','trestbps','chol','fbs'],'risk4':['age','sex','trestbps','fbs']}

def nested(d,features,name,jobs=1,outer=5,operating=False):
    d=d.reset_index(drop=True); rows=[]; fits=[]; rules=[]
    for f,(tr,te) in enumerate(split(d,outer,2026)):
        a,b=d.iloc[tr],d.iloc[te]; m,params=tune(a,features,name,400+f,jobs)
        p=m.predict_proba(b[features])[:,1]; r=pd.DataFrame(dict(row=te,fold=f,target=b.target.to_numpy(),probability=p));r['parameters']=json.dumps(params);r['predictor_overlap_count']=overlap_count(a,b)
        if operating:
            q=np.empty(len(a))
            for j,(u,v) in enumerate(split(a,3,400+f)):
                inner,_=tune(a.iloc[u],features,name,400+f,jobs);q[v]=inner.predict_proba(a.iloc[v][features])[:,1]
            fpr,tpr,cut=roc_curve(a.target,q,drop_intermediate=False)
            choices={'fixed':.5,'youden':cut[np.argmax(tpr-fpr)],'sensitivity90':max(cut[tpr>=.9]),'sensitivity95':max(cut[tpr>=.95])}
            r['recalibrated']=fit_cal(a.target,q).predict_proba(logits(p))[:,1]
            for rule,t in choices.items():
                tn,fp,fn,tp=confusion_matrix(b.target,p>=t,labels=[0,1]).ravel();rules.append(dict(fold=f,rule=rule,threshold=float(t),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn)))
        rows.append(r);fits.append((a,b,m))
    return pd.concat(rows).sort_values('row'),fits,pd.DataFrame(rules)

def summary(p):
    a=pd.DataFrame([metrics(g.target,g.probability) for _,g in p.groupby('fold')]);return dict(mean_auroc=a.auroc.mean(),sd_auroc=a.auroc.std(),accuracy=a.accuracy.mean(),pooled_auroc=roc_auc_score(p.target,p.probability),ci=ci(p.target,p.probability))

def duplicate_test(comp):
    eligible=set(map(tuple,comp.loc[comp.duplicated(KEY,keep=False),KEY].to_numpy()));d=comp.drop_duplicates(KEY).reset_index(drop=True);rows=[]
    for seed in range(100,120):
        tr,te=train_test_split(np.arange(len(d)),test_size=.2,stratify=d.target,random_state=seed);a,b=d.iloc[tr],d.iloc[te];extra=b.loc[[tuple(r) in eligible for r in b[KEY].to_numpy()]]
        ctrl=pd.concat([a[a.target==y].sample(len(g),replace=False,random_state=seed) for y,g in extra.groupby('target')]) if len(extra) else a.iloc[:0]; r={'seed':seed,'copies':len(extra)}
        for label,add in [('control',ctrl),('overlap',extra)]:
            train=clean(pd.concat([a,add]));test=clean(b);m=pipe(FEATURES,XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.1,random_state=2026,n_jobs=1,eval_metric='logloss')).fit(train[FEATURES],train.target);p=m.predict_proba(test[FEATURES])[:,1];r.update({label+'_'+k:v for k,v in metrics(b.target,p).items()})
        r['delta_auroc']=r['overlap_auroc']-r['control_auroc'];r['delta_accuracy']=r['overlap_accuracy']-r['control_accuracy'];rows.append(r)
    return pd.DataFrame(rows)

def esc_prob(r):
    table=[[3,5,4,3,1,1],[22,10,10,6,3,2],[32,13,17,6,11,3],[44,16,26,11,22,6],[52,27,34,19,24,10]]
    if pd.isna(r.age) or r.age<30 or r.cp not in [1,2,3] or r.sex not in [0,1]: return np.nan
    return table[min(int(r.age)//10-3,4)][2*(int(r.cp)-1)+(1-int(r.sex))]/100

def explain(fits,out):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    ranks=[];local=[];comparison=[]
    for f,(a,b,m) in enumerate(fits):
        prep=m.named_steps['prep'];clf=m.named_steps['clf'];x=prep.transform(a[FEATURES]);z=prep.transform(b[FEATURES]);names=prep.get_feature_names_out();mapping=np.array([next(c for c in FEATURES if n.split('__',1)[1]==c or n.split('__',1)[1].startswith(c+'_')) for n in names])
        def agg(v): return np.column_stack([v[:,mapping==c].sum(1) for c in FEATURES])
        sv=agg(np.asarray(shap.TreeExplainer(clf,feature_perturbation='tree_path_dependent').shap_values(z)));ranks.append(np.abs(sv).mean(0))
        if f!=0: continue
        bg=x[np.random.default_rng(901).choice(len(x),80,replace=False)];iv=agg(np.asarray(shap.TreeExplainer(clf,data=bg,feature_perturbation='interventional').shap_values(z)));comparison=pd.DataFrame({'feature':FEATURES,'path':ranks[0],'interventional':np.abs(iv).mean(0)})
        sentinel=-999.;raw=a[FEATURES].fillna(sentinel).to_numpy(float);cat=[i for i,c in enumerate(FEATURES) if c not in NUMERIC]; lime=LimeTabularExplainer(raw,feature_names=FEATURES,categorical_features=cat,mode='classification',discretize_continuous=True,discretizer='quartile',random_state=902)
        def predict(v): return m.predict_proba(pd.DataFrame(v,columns=FEATURES).replace(sentinel,np.nan))
        for i in np.random.default_rng(903).choice(len(b),100,replace=False):
            e=lime.explain_instance(b[FEATURES].iloc[i].fillna(sentinel).to_numpy(float),predict,labels=(1,),num_features=11,num_samples=1500);v=np.zeros(11)
            for j,w in e.local_exp[1]: v[j]=w
            local.append(dict(row=int(i),spearman=float(spearmanr(abs(sv[i]),abs(v)).statistic),overlap=overlap(sv[i],v),fidelity=e.score))
    comparison.to_csv(out/'shap_comparison.csv',index=False);pd.DataFrame(local).to_csv(out/'lime_agreement.csv',index=False);pd.DataFrame([dict(a=i,b=j,spearman=spearmanr(ranks[i],ranks[j]).statistic,overlap=overlap(ranks[i],ranks[j])) for i,j in combinations(range(5),2)]).to_csv(out/'shap_stability.csv',index=False)

def overlap(a,b): return len(set(np.argsort(-abs(a))[:3])&set(np.argsort(-abs(b))[:3]))/3

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--output',type=Path,default=Path('results'));p.add_argument('--jobs',type=int,default=1);p.add_argument('--skip-explanations',action='store_true');args=p.parse_args();o=args.output;
    if o.exists() and any(o.iterdir()): raise SystemExit('Output directory is not empty. Select a new --output directory.')
    if not args.skip_explanations:
        import shap
        import lime
    o.mkdir(parents=True,exist_ok=True)
    raw,comp,stat=load_data(args.data,'one-based')
    if (len(raw),len(comp),len(stat))!=(920,1190,270): raise ValueError('Input row counts differ from the R4 datasets')
    au=audit(raw,comp,stat,o)
    d=clean(raw.drop_duplicates(KEY).reset_index(drop=True));c=comp.drop_duplicates(KEY).reset_index(drop=True)
    if len(d)!=918 or len(c)!=918: raise ValueError('Exact deduplication did not produce 918 observations in each representation')
    write_json(o/'manifest.json',{'hashes':{f:hashlib.sha256((args.data/f).read_bytes()).hexdigest() for f in list(SOURCES.values())+['heart.dat','heart_disease_uci.csv']},'versions':{k:version(k) for k in ['numpy','pandas','scikit-learn','xgboost','scipy']+([] if args.skip_explanations else ['shap','lime'])},'configuration':json.loads((Path(__file__).parent/'config/experiments.json').read_text())})
    primary={};cal={};abl=[];sens=[]
    for name in models():
        pred,fits,rules=nested(d,FEATURES,name,args.jobs,operating=name=='XGB');primary[name]=summary(pred);pred.to_csv(o/f'primary_{name}.csv',index=False)
        if name in ['LR','RF','XGB']: cal[name]=calibration(pred.target,pred.probability);abl.append(dict(model=name,subset='all11',**primary[name]))
        if name=='XGB': xp=pred;xf=fits;rules.to_csv(o/'threshold_folds.csv',index=False);cal['XGB_recalibrated']=calibration(pred.target,pred.recalibrated)
    write_json(o/'primary.json',primary);write_json(o/'calibration.json',cal)
    for name in ['LR','RF','XGB']:
        for k,fs in SETS.items():
            if k!='all11': q,_,_=nested(d,fs,name,args.jobs);abl.append(dict(model=name,subset=k,**summary(q)))
    pd.DataFrame(abl).to_csv(o/'ablation.csv',index=False)
    for label,frame in [('original',raw.drop_duplicates(KEY)),('compilation',c)]:
        for mode in ['zero','median','indicator','omitted']:
            fs=FEATURES+['chol_missing'] if mode=='indicator' else [f for f in FEATURES if f!='chol'] if mode=='omitted' else FEATURES
            q,_,_=nested(clean(frame,mode),fs,'XGB',args.jobs);sens.append(dict(dataset=label,treatment=mode,**summary(q)))
    pd.DataFrame(sens).to_csv(o/'cholesterol.csv',index=False)
    source=[]
    for s in d.source.unique():
        a=d[d.source!=s];b=d[d.source==s];m,_=tune(a,FEATURES,'XGB',400,args.jobs);v=m.predict_proba(b[FEATURES])[:,1];q,_,_=nested(b,FEATURES,'XGB',args.jobs,outer=4);source.append(dict(source=s,n=len(b),positive=int(b.target.sum()),heldout=roc_auc_score(b.target,v),ci=ci(b.target,v),within=summary(q)['mean_auroc']))
    pd.DataFrame(source).to_csv(o/'sources.csv',index=False)
    sp=np.empty(len(d),object)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=2026).split(d,d.source):
        m=pipe(FEATURES,LogisticRegression(C=1,max_iter=5000)).fit(d.iloc[tr][FEATURES],d.iloc[tr].source);sp[te]=m.predict(d.iloc[te][FEATURES])
    write_json(o/'source_probe.json',{'balanced_accuracy':balanced_accuracy_score(d.source,sp)})
    sub=[]
    for label,mask in [('women',d.sex==0),('men',d.sex==1),('chol_unavailable',d.chol.isna()),('chol_recorded',d.chol.notna())]:
        q=xp.loc[mask.to_numpy()];sub.append(dict(group=label,n=len(q),positive=int(q.target.sum()),auroc=roc_auc_score(q.target,q.probability),ci=ci(q.target,q.probability)))
    pd.DataFrame(sub).to_csv(o/'subgroups.csv',index=False)
    q,_,_=nested(d,['age','sex','cp'],'LR',args.jobs);e=d.apply(esc_prob,axis=1);mask=e.notna();q=q.loc[mask.to_numpy()];write_json(o/'esc.json',{'n':len(q),'positive':int(q.target.sum()),'esc_auroc':roc_auc_score(q.target,e[mask]),'esc_ci':ci(q.target,e[mask]),'lr_auroc':roc_auc_score(q.target,q.probability),'lr_ci':ci(q.target,q.probability)})
    duplicate_test(comp).to_csv(o/'duplicates.csv',index=False)
    rr=[]
    for rule,g in rules.groupby('rule'):
        tp,fp,tn,fn=g[['tp','fp','tn','fn']].sum();rr.append(dict(rule=rule,median=g.threshold.median(),minimum=g.threshold.min(),maximum=g.threshold.max(),tp=tp,fp=fp,tn=tn,fn=fn,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),ppv=tp/(tp+fp) if tp+fp else np.nan,npv=tn/(tn+fn) if tn+fn else np.nan))
    pd.DataFrame(rr).to_csv(o/'thresholds.csv',index=False)
    if not args.skip_explanations: explain(xf,o)
    from descriptives import export_tables
    export_tables(raw,comp,stat,d,xp,rules,o)
    d.groupby('source')[FEATURES].agg(lambda x:x.isna().sum()).to_csv(o/'missingness.csv');d.groupby('source').agg(n=('target','size'),positive=('target','sum')).to_csv(o/'cohorts.csv')
    write_json(o/'metrics.json',{'primary':primary,'calibration':cal,'audit':au,'explanations_executed':not args.skip_explanations,'predictor_overlap_by_fold':xp.groupby('fold').predictor_overlap_count.first().to_dict()})
if __name__=='__main__': main()
