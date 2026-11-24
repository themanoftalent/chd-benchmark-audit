from common import *
ANCHORS=['age','sex','cp','trestbps','thalch','exang']

def audit(raw,comp,stat,out):
    ref=pd.concat([raw,stat],ignore_index=True); ref['family']=ref.source.replace({'Statlog':'Cleveland'})
    a=ref[FEATURES].to_numpy(float); obs=~np.isnan(a); ac=obs[:,[FEATURES.index(c) for c in ANCHORS]].sum(1)
    links=[]; fills=[]; sensitivity=[]
    for minimum in [3,4,5,6]:
        counts=dict(anchors=minimum,candidates=0,one_family=0,multiple_families=0,unmatched=0,discordant=0)
        for i,r in comp.iterrows():
            b=r[FEATURES].to_numpy(float); hit=ref.loc[((a==b)|~obs).all(1)&(ac>=minimum)]
            fam=hit.family.unique(); discord=bool((hit.target!=r.target).any()); counts['discordant']+=discord
            counts['candidates']+=bool(len(hit)); counts['one_family']+=len(fam)==1;counts['multiple_families']+=len(fam)>1;counts['unmatched']+=not len(hit)
            if minimum!=5: continue
            links.append(dict(row=i,family=fam[0] if len(fam)==1 else '',candidate_count=len(hit),candidates=';'.join(hit.record_id),discordant=discord))
            if len(fam)==1:
                for c in ['slope','fbs','oldpeak','chol']:
                    if hit[c].isna().all() and pd.notna(r[c]): fills.append(dict(row=i,family=fam[0],field=c,target=r.target,value=r[c]))
        sensitivity.append(counts)
    pd.DataFrame(links).to_csv(out/'row_linkage.csv',index=False);pd.DataFrame(fills,columns=['row','family','field','target','value']).to_csv(out/'fills.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'anchor_sensitivity.csv',index=False)
    near=[]; d=comp.drop_duplicates(KEY); cat=[c for c in KEY if c not in NUMERIC]; tol=np.array([1,1,1,1,.1]);
    for mult,k in [(1,1),(2,1),(5,1),(2,2),(5,2)]:
        pairs=[]
        for _,g in d.groupby(cat,dropna=False):
            arr=g[NUMERIC].to_numpy(float); ids=g.index.to_numpy()
            for i in range(len(g)):
                delta=np.abs(arr[i+1:]-arr[i]); changed=(delta>1e-8).sum(1)
                ok=(changed>0)&(changed<=k)&(delta<=tol*mult+1e-8).all(1)&~((arr[i,2]==0)!=(arr[i+1:,2]==0))
                pairs.extend((int(ids[i]),int(j)) for j in ids[i+1:][ok])
        near.append(dict(multiplier=mult,fields=k,pairs=len(pairs),unique_rows=len(set(sum(([i,j] for i,j in pairs),[])))))
    pd.DataFrame(near).to_csv(out/'near_duplicates.csv',index=False)
    summary={}
    for name,d in [('original',raw),('compilation',comp)]:
        summary[name]={'rows':len(d),'duplicates':int(d.duplicated(KEY).sum()),'predictor_duplicates':int(d.duplicated(FEATURES).sum()),'conflicting_vectors':int((d.groupby(FEATURES,dropna=False).target.nunique()>1).sum())}
    write_json(out/'audit.json',summary)
    return summary
