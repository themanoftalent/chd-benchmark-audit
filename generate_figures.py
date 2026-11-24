"""Render manuscript figures from generated CSV outputs, plus the source workflow."""
from pathlib import Path
import argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def save(fig,out,name):
    for ext in ['png','svg']: fig.savefig(out/f'{name}.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def workflow(out):
    fig,ax=plt.subplots(figsize=(10,7));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,t,fc):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=fc,edgecolor='#334155',linewidth=1));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10,linespacing=1.35)
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.3,color='#334155'))
    box(.15,7.7,3,1.65,'Distributed compilation\n1,190 records\n272 exact repeats','#e8eff8')
    box(3.45,7.7,3.1,1.65,'Four original UCI cohorts (920)\nCleveland 303 | Hungarian 294\nSwitzerland 123 | Long Beach VA 200','#eaf3ee')
    box(6.85,7.7,3,1.65,'Statlog (270 records)\nProvenance audit only\nOverlaps Cleveland','#f4f0e7')
    box(.15,5.45,4.4,1.45,'Source linkage and duplicate audit\nFive files, 1,190 reference rows\nOriginal missing-value comparison','#e8eff8')
    box(5.35,5.45,4.5,1.45,'Original UCI primary dataset\n920 − 2 exact repeats = 918\n11 predictors, binary disease label','#eaf3ee')
    arrow((1.65,7.7),(1.65,6.9));arrow((4.2,7.7),(3.7,6.9));arrow((5.8,7.7),(6.9,6.9));arrow((8.35,7.7),(4.45,6.85))
    box(.15,3.0,4.4,1.65,'Compilation sensitivity experiments\n918 after exact deduplication\n20 matched duplicate contrasts\nFour cholesterol treatments','#e8eff8')
    box(5.35,3.0,4.5,1.65,'Nested cross-validation\n5 outer folds | 3 inner folds\n6 classifiers | 8 candidates each\nTraining-only preprocessing and tuning','#eaf3ee')
    arrow((2.35,5.45),(2.35,4.65));arrow((7.6,5.45),(7.6,4.65))
    box(.15,.5,4.4,1.65,'Comparative assessment\nDuplicate inflation and source sensitivity\nDiagnostic-stage ablations','#f4f0e7')
    box(5.35,.5,4.5,1.65,'Evaluation on outer assessment folds\nDiscrimination, calibration and thresholds\nSubgroups and SHAP/LIME agreement','#f4f0e7')
    arrow((2.35,3),(2.35,2.15));arrow((7.6,3),(7.6,2.15));save(fig,out,'Figure_1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--output',type=Path,default=Path('figures'));p.add_argument('--workflow-only',action='store_true');a=p.parse_args();r=a.results;o=a.output;o.mkdir(parents=True,exist_ok=True);workflow(o)
    if a.workflow_only:return
    d=pd.read_csv(r/'cholesterol.csv');t=['zero','median','indicator','omitted'];labels=['Zero retained','Fold median','Median + indicator','Omitted'];x=np.arange(4);fig,ax=plt.subplots(figsize=(7,4));
    for i,(g,c) in enumerate([('original','#456c98'),('compilation','#b77949')]):
        q=d[d.dataset==g].set_index('treatment').loc[t];ax.bar(x+(i-.5)*.36,q.mean_auroc,.36,label=g.capitalize(),color=c)
    ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,1);ax.set_ylabel('Mean outer-fold AUROC');ax.legend(frameon=False);save(fig,o,'Figure_2')
    m=pd.read_csv(r/'A5_missingness.csv',index_col=0);n=pd.read_csv(r/'A4_cohorts.csv').set_index('source').retained;v=m.div(n,axis=0)*100;fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(v.to_numpy(),aspect='auto',vmin=0,vmax=100,cmap='Blues');ax.set_xticks(range(len(v.columns)),v.columns,rotation=40,ha='right');ax.set_yticks(range(len(v)),v.index);fig.colorbar(im,ax=ax,label='Unavailable (%)');save(fig,o,'Figure_A1')
    d=pd.read_csv(r/'duplicates.csv');fig,axes=plt.subplots(2,1,figsize=(7,5),sharex=True)
    for ax,c,scale,l in [(axes[0],'delta_auroc',1,'Δ AUROC'),(axes[1],'delta_accuracy',100,'Δ accuracy (pp)')]:
        y=d[c]*scale;ax.plot(d.seed,y,'o-',color='#456c98',lw=1);ax.axhline(y.mean(),ls='--',color='gray');ax.set_ylabel(l)
    axes[1].set_xlabel('Matched partition seed');save(fig,o,'Figure_A2')
    d=pd.read_csv(r/'ablation.csv');t=['all11','no_exercise','risk5','risk4'];fig,ax=plt.subplots(figsize=(7,4));
    for name,g in d.groupby('model'):ax.plot(range(4),g.set_index('subset').loc[t].mean_auroc,'o-',label=name)
    ax.set_xticks(range(4),['All 11','No exercise','Risk factors','No cholesterol'],rotation=15);ax.set_ylabel('Mean outer-fold AUROC');ax.set_ylim(0,1);ax.legend(frameon=False);save(fig,o,'Figure_A3')
    import ast
    d=pd.read_csv(r/'sources.csv');interval=np.array([ast.literal_eval(v) for v in d.ci]);x=np.arange(len(d));fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(x-.08,d.heldout,yerr=[d.heldout-interval[:,0],interval[:,1]-d.heldout],fmt='o',capsize=4,label='Source-held-out');ax.plot(x+.08,d.within,'s',label='Within-source');ax.set_xticks(x,d.source,rotation=15);ax.set_ylim(0,1);ax.set_ylabel('AUROC');ax.legend(frameon=False);save(fig,o,'Figure_A4')
    d=pd.read_csv(r/'thresholds.csv');fig,ax=plt.subplots(figsize=(6,5));ax.scatter(1-d.specificity,d.sensitivity,color='#456c98');
    for _,q in d.iterrows():ax.annotate(q.rule,(1-q.specificity,q.sensitivity),xytext=(5,-10),textcoords='offset points',fontsize=9)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive rate',ylabel='Sensitivity');save(fig,o,'Figure_A5')
if __name__=='__main__':main()
