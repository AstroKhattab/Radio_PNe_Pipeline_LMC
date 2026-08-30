"""Compare the radio PNLF under three masking/sample choices.

Panels: the full sample unmasked, the full sample with the published
bright-bin and faint-cut mask, and the high-confidence sample unmasked.

Input   03_Outputs/step6a_pnlf.vot, 03_Outputs/step5b_spectral_index_recut.vot
Output  05_Figures/step6e_pnlf_mask_vs_clean.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import warnings, numpy as np, os; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.table import Table
from scipy.optimize import differential_evolution, curve_fit
np.random.seed(42)
B=os.environ.get("PNBASE",".")
t=Table.read(f'{B}/03_Outputs/step6a_pnlf.vot',format='votable')
c=Table.read(f'{B}/03_Outputs/step5b_spectral_index_recut.vot',format='votable')
M=np.array(t['M_radio'],dtype=float); idt=np.array([str(x) for x in t['RP_ID']])
idc={str(k):i for i,k in enumerate(c['RP_ID'])}
J=np.array([idc.get(k,-1) for k in idt])
oc=np.array([str(c['our_classification'][j]) if j>=0 else '?' for j in J])
sp=np.array([str(c['sp_class_recut'][j]) if j>=0 else '?' for j in J])
fin=np.isfinite(M); clean=fin&np.isin(oc,['High-confidence PN','Probable PN'])
BIN=.3; MLIM=-.5; ALO,AHI=-4.4,-3.7

def cc(m,N,Ms):
    v=N*np.exp(0.307*(m-Ms))*(1-np.exp(3.0*(Ms-m))); return np.where(m>Ms,np.maximum(v,1e-4),1e-4)
def gau(m,A,mu,s): return A*np.exp(-0.5*((m-mu)/s)**2)
def bins(sel):
    m=M[sel]; e=np.arange(np.floor(m.min()/BIN)*BIN,np.ceil(m.max()/BIN)*BIN+BIN,BIN)
    cn,_=np.histogram(m,bins=e); ct=.5*(e[:-1]+e[1:]); nz=cn>0
    return ct[nz],cn[nz].astype(float),e
def fit(fn,mx,my,lb,ub):
    ye=np.sqrt(my); ye[ye==0]=1
    r=lambda p: np.sum(((my-fn(mx,*p))/ye)**2)
    de=differential_evolution(r,bounds=list(zip(lb,ub)),seed=42,maxiter=9000,tol=1e-12,popsize=35)
    po,_=curve_fit(fn,mx,my,p0=de.x,bounds=(lb,ub),sigma=ye,absolute_sigma=True,maxfev=200000)
    yf=fn(mx,*po); k=len(po); ch=np.sum(((my-yf)/ye)**2)
    return po, ch+2*k, ch/(len(my)-k)

fig,axes=plt.subplots(1,3,figsize=(14.2,4.5),sharey=True)
COL={'thermal':'#1f6fb4','uncertain':'#e59a1e','steep':'#c0392b','no_fit':'0.72'}

# panel A: all sources, unmasked
ax=axes[0]; mx,my,e=bins(fin); keep=mx<=MLIM
bot=np.zeros(len(mx))
for s in ['thermal','uncertain','no_fit','steep']:
    h,_=np.histogram(M[fin&(sp==s)],bins=e); h=h[np.histogram(M[fin],bins=e)[0]>0]
    ax.bar(mx,h,bottom=bot,width=BIN*0.92,color=COL[s],lw=0,label=s.replace('_',' '))
    bot+=h
po,aic,rc=fit(cc,mx[keep],my[keep],[0,np.nanmin(M)-5],[1e6,M[fin].min()-0.05])
g=np.linspace(po[1]+.01,MLIM,500); ax.plot(g,cc(g,*po),'k-',lw=2)
ax.axvline(MLIM,ls=':',color='.4'); ax.set_yscale('log'); ax.set_ylim(.5,140)
# ax.set_title("A.  All 220, no mask\nCiardullo canonical: $\\Delta$AIC = 22.7  (rank 9/10)",fontsize=10)
# changed 2026-08-30: the AIC and the rank were frozen strings that a refit could not update;
# the title is now filled in after the three fits (see set_title call below).  The rank was
# dropped because this script fits one model per panel and cannot rank ten of them.
title_A = ax.set_title("",fontsize=10)
ax.set_ylabel("$N$ per 0.3 mag bin"); ax.legend(fontsize=7.5,frameon=False,loc='upper left')
ax.annotate("brightest source\nis steep-spectrum",xy=(-4.9,1.1),xytext=(-4.7,14),fontsize=7.8,
            color='#c0392b',arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1))

# panel B: all sources, published mask
ax=axes[1]; keep2=(mx<=MLIM)&~((mx>ALO)&(mx<AHI))
bot=np.zeros(len(mx))
for s in ['thermal','uncertain','no_fit','steep']:
    h,_=np.histogram(M[fin&(sp==s)],bins=e); h=h[np.histogram(M[fin],bins=e)[0]>0]
    ax.bar(mx,h,bottom=bot,width=BIN*0.92,color=COL[s],lw=0,alpha=.35); bot+=h
bot=np.zeros(int(keep2.sum()))
for s in ['thermal','uncertain','no_fit','steep']:
    h,_=np.histogram(M[fin&(sp==s)],bins=e); h=h[np.histogram(M[fin],bins=e)[0]>0]
    ax.bar(mx[keep2],h[keep2],bottom=bot,width=BIN*0.92,color=COL[s],lw=0); bot+=h[keep2]
po2,aic2,rc2=fit(cc,mx[keep2],my[keep2],[0,np.nanmin(M)-5],[1e6,M[fin].min()-0.05])
g=np.linspace(po2[1]+.01,MLIM,500); ax.plot(g,cc(g,*po2),'k-',lw=2)
ax.axvspan(ALO,AHI,color='k',alpha=.07)
ax.axvline(MLIM,ls=':',color='.4'); ax.set_yscale('log')
ax.set_title(f"B.  All 220, bright bins masked (published)\n$M^*$ = {po2[1]:.3f}  —  anchored by an AGN",fontsize=10)
# added 2026-08-30: the ghosted bars reach N ~ 100 with nothing saying they are masked
ax.legend(handles=[Patch(facecolor="0.55",alpha=.35,
                         label="Masked bins (plotted, not fitted)")],
          fontsize=7.5,frameon=False,loc="upper left")
ax.set_xlabel("$M_{\\rm radio}$ / mag")

# panel C: clean sample, unmasked
ax=axes[2]; mxc,myc,ec=bins(clean); keepc=mxc<=MLIM
bot=np.zeros(len(mxc))
for s in ['thermal','uncertain','no_fit','steep']:
    h,_=np.histogram(M[clean&(sp==s)],bins=ec); h=h[np.histogram(M[clean],bins=ec)[0]>0]
    ax.bar(mxc,h,bottom=bot,width=BIN*0.92,color=COL[s],lw=0); bot+=h
poc,aicc,rcc=fit(cc,mxc[keepc],myc[keepc],[0,np.nanmin(M)-5],[1e6,M[clean].min()-0.05])
g=np.linspace(poc[1]+.01,MLIM,500); ax.plot(g,cc(g,*poc),'k-',lw=2,label='Ciardullo canonical')
ax.axvline(MLIM,ls=':',color='.4'); ax.set_yscale('log')
# ax.set_title(f"C.  118 High-confidence + Probable, no mask\n$M^*$ = {poc[1]:.3f}   $\\Delta$AIC = 0.10  (rank 2/10)",fontsize=10)
# changed 2026-08-30: see the note on panel A; delta-AIC is now computed, the rank dropped.
title_C = ax.set_title("",fontsize=10)

# Panel B, the published masked fit, is the reference the other two are quoted against.
dAIC_A = aic - aic2
dAIC_C = aicc - aic2
title_A.set_text("A.  All 220, no mask\n"
                 f"Ciardullo canonical: $\\Delta$AIC = {dAIC_A:.1f} vs panel B")
title_C.set_text("C.  118 High-confidence + Probable, no mask\n"
                 f"$M^*$ = {poc[1]:.3f}   $\\Delta$AIC = {dAIC_C:.1f} vs panel B")
ax.legend(fontsize=8,frameon=False)
for a in axes:
    a.set_xlim(-5.5,2.3)
    for s in ('top','right'): a.spines[s].set_visible(False)
fig.suptitle("Radio PNLF — effect of luminosity masking versus sample cleaning",fontsize=12.5,y=1.0)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(f'{B}/05_Figures/step6e_pnlf_mask_vs_clean.pdf')
print("A  M*=%.3f  chi2red=%.2f"%(po[1],rc))
print("B  M*=%.3f  chi2red=%.2f"%(po2[1],rc2))
print("C  M*=%.3f  chi2red=%.2f"%(poc[1],rcc))
print("saved step6e_pnlf_mask_vs_clean.pdf")
