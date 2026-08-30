"""Single summary figure: detection fraction, evidence grade and spectral class.

Replaces the separate count panels used in earlier drafts.

Input   03_Outputs/step05a_spectral_index_recut.vot
Output  04_Figures/step08a_detection_and_evidence.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import warnings, os; warnings.filterwarnings('ignore')
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

B=os.environ.get("PNBASE",".")
c=Table.read(B+'/03_Outputs/step05a_spectral_index_recut.vot',format='votable')
S=lambda k: np.array([str(x).strip() for x in c[k]])
rc,surv,oc,sp,hs=S('reid_classification'),S('detection_surveys'),S('our_classification'),S('sp_class_recut'),S('hash_status_label')
TOT={'True':311,'Known':275,'Likely':40,'Possible':47,'Unknown':6}

def wilson(k,n,z=1.0):
    if n==0: return 0.,0.,0.
    p=k/n; d=1+z*z/n; ctr=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0,ctr-half),min(1,ctr+half)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.9,
                     'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
fig,axes=plt.subplots(1,3,figsize=(11.6,3.5))
CB={'thermal':'#1f6fb4','uncertain':'#e59a1e','steep':'#c0392b','no_fit':'#b9c0c8'}

# --- (a) detection fraction vs Reid class, with Wilson intervals
ax=axes[0]
classes=['True','Known','Likely','Possible']
x=np.arange(len(classes))
for off,(lab,mask,mk) in enumerate([
    ('MeerKAT', (surv=='MeerKAT')|(surv=='both'), 'o'),
    ('ASKAP',   (surv=='ASKAP')|(surv=='both'),  's'),
    ('Union',   np.ones(len(rc),bool),           'D')]):
    p=[];lo=[];hi=[]
    for k in classes:
        m=(rc==k)&mask
        a,b_,c_=wilson(int(m.sum()),TOT[k]); p.append(100*a); lo.append(100*(a-b_)); hi.append(100*(c_-a))
    ax.errorbar(x+(off-1)*0.16, p, yerr=[lo,hi], fmt=mk, ms=4.6, lw=1.1, capsize=2.4, label=lab)
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylabel('Radio detection fraction (per cent)')
ax.set_xlabel('R14 optical class')
# ax.set_ylim(-3,78); ax.legend(frameon=False, fontsize=8, loc='upper left')
# changed 2026-08-30: this is a percentage axis, so it should not run below zero
ax.set_ylim(0,78); ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.text(0.97,0.95,'(a)',transform=ax.transAxes,ha='right',va='top',fontweight='bold')

# --- (b) spectral composition of each confidence grade
ax=axes[1]
grades=['High-confidence PN','Probable PN','Possible PN','Questionable/review']
# short=['High','Probable','Possible','Question.']
# changed 2026-08-30: 'Question.' was a truncated word, not an abbreviation
short=['High','Probable','Possible','Questionable']
order=['thermal','uncertain','steep','no_fit']
labels={'thermal':'Thermal','uncertain':'Uncertain','steep':'Steep','no_fit':'No reliable $\\alpha$'}
y=np.arange(len(grades)); left=np.zeros(len(grades))
for s in order:
    frac=np.array([np.sum((oc==g)&(sp==s))/max(np.sum(oc==g),1) for g in grades])*100
    ax.barh(y,frac,left=left,color=CB[s],height=0.62,label=labels[s],edgecolor='white',lw=0.6)
    left+=frac
for i,g in enumerate(grades):
    ax.text(101.5,y[i],f'$N$={int(np.sum(oc==g))}',va='center',fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(short); ax.invert_yaxis()
ax.set_xlabel('Spectral class (per cent of grade)'); ax.set_xlim(0,100)
ax.legend(frameon=False,fontsize=7.4,ncol=2,loc='lower center',bbox_to_anchor=(0.5,-0.42))
ax.text(0.97,0.95,'(b)',transform=ax.transAxes,ha='right',va='top',fontweight='bold')

# --- (c) MIR ratio distribution by spectral class (the diagnostic that matters)
ax=axes[2]
R=np.array(c['cohen_ratio_near1GHz'],dtype=float)
bins=np.logspace(np.log10(0.05),np.log10(3000),26)
for s in ['thermal','uncertain','steep']:
    m=(sp==s)&np.isfinite(R)
    ax.hist(R[m],bins=bins,histtype='step',lw=1.6,color=CB[s],label=f'{labels[s]} ($N$={int(m.sum())})')
ax.axvspan(0.5,10,color='0.85',zorder=0)
ax.axvline(4.7,ls='--',lw=1.1,color='0.35')
ax.set_xscale('log'); ax.set_xlabel(r'$R = F_{8\,\mu\mathrm{m}}/S_{\rm radio}$')
ax.set_ylabel('Number of sources')
ax.legend(frameon=False,fontsize=7.6,loc='upper right')
ax.text(0.03,0.95,'(c)',transform=ax.transAxes,ha='left',va='top',fontweight='bold')
# ax.text(0.5,0.02,'Galactic band',transform=ax.transAxes,ha='center',fontsize=7,color='0.4')
# changed 2026-08-30: darkened for legibility against the 0.85 grey span and placed at the
# geometric centre of the 0.5-10 band it labels (x in data units, y in axes fraction)
ax.text(np.sqrt(0.5*10),0.03,'Galactic band',transform=ax.get_xaxis_transform(),
        ha='center',va='bottom',fontsize=7,color='0.25')

for a in axes:
    for s in ('top','right'): a.spines[s].set_visible(True)
fig.tight_layout(pad=0.7)
out=B+"/04_Figures/step08a_detection_and_evidence.pdf"
fig.savefig(out); print("wrote",out)
