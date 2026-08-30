"""
step07g_pnlf_grids_by_sample.py
Radio PNLF empirical-model grids under several sample / masking scenarios.
Vertical 2 x 5 layout sized for a full A4 portrait page in Overleaf.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import warnings, os, numpy as np; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from scipy.optimize import differential_evolution, curve_fit
from scipy.stats import t as student_t, skewnorm
np.random.seed(42)

B=os.environ.get("PNBASE",".")
t=Table.read(f'{B}/03_Outputs/step07a_pnlf.vot',format='votable')
c=Table.read(f'{B}/03_Outputs/step05a_spectral_index_recut.vot',format='votable')
M=np.array(t['M_radio'],dtype=float); idt=np.array([str(x) for x in t['RP_ID']])
idc={str(k):i for i,k in enumerate(c['RP_ID'])}
J=np.array([idc.get(k,-1) for k in idt])
oc=np.array([str(c['our_classification'][j]) if j>=0 else '?' for j in J])
sp=np.array([str(c['sp_class_recut'][j]) if j>=0 else '?' for j in J])
fin=np.isfinite(M)
BIN=0.3; MLIM=-0.5; ALO,AHI=-4.4,-3.7

SCEN=[
 ("S1_ALL_published",  fin,                         True,  True,  "All 220 — published mask\n(bright bins + faint cut)"),
 ("S2_ALL_faintonly",  fin,                         False, True,  "All 220 — faint cut only\n(no bright mask)"),
 ("S3_ALL_nomask",     fin,                         False, False, "All 220 — NO masking at all\n(every non-empty bin fitted)"),
 ("S4_NONSTEEP",       fin&(sp!='steep'),           False, True,  "Steep spectra removed (N=%d)\nfaint cut only"),
 ("S5_CLEAN",          fin&np.isin(oc,['High-confidence PN','Probable PN']), False, True,
                                                                   "High-confidence + Probable (N=%d)\nfaint cut only"),
]

def build(sel,mask_bright,faint_cut):
    m=M[sel]
    e=np.arange(np.floor(m.min()/BIN)*BIN,np.ceil(m.max()/BIN)*BIN+BIN,BIN)
    cn,_=np.histogram(m,bins=e); ct=0.5*(e[:-1]+e[1:]); nz=cn>0
    mx=ct[nz]; my=cn[nz].astype(float); ye=np.sqrt(my); ye[ye==0]=1.0
    keep=np.ones(len(mx),bool)
    if faint_cut:   keep &= ~(mx>MLIM)
    if mask_bright: keep &= ~((mx>ALO)&(mx<AHI))
    return mx,my,ye,keep

def MODELS(mx,my,mmin):
    pk=mx[np.argmax(my)]; hi=my.max(); Msx=mmin-0.05; Msn=mmin-5.0
    f={}
    f['Gaussian']=(lambda m,A,mu,s: A*np.exp(-0.5*((m-mu)/s)**2),[0,mx.min()-3,.1],[hi*4,mx.max()+2,8])
    f['Student-$t$']=(lambda m,A,mu,s,nu: A*student_t.pdf(m,max(nu,.5),loc=mu,scale=max(abs(s),.01)),[0,mx.min()-3,.1,.5],[hi*3,mx.max()+2,10,200])
    f['Cauchy (Lorentzian)']=(lambda m,A,mu,g: A/(1+((m-mu)/g)**2),[0,mx.min()-3,.05],[hi*4,mx.max()+2,10])
    f['Skew-normal']=(lambda m,A,l,s,a: A*skewnorm.pdf(m,a,loc=l,scale=s),[0,mx.min()-4,.1,-30],[hi*6,mx.max()+3,15,30])
    f['Polynomial (2)']=(lambda m,a,b,cc_: a*m**2+b*m+cc_,[-500,-200,0],[500,200,hi*3])
    f['Polynomial (3)']=(lambda m,a,b,cc_,d: a*m**3+b*m**2+cc_*m+d,[-500,-500,-200,0],[500,500,200,hi*3])
    f['Exponential']=(lambda m,a,b,cc_: a*np.exp(b*m)+cc_,[0,0,-50],[hi*10,5,hi])
    def sch(m,p,ms,al):
        d=m-ms; return np.maximum(p*10**(0.4*(al+1)*d)*np.exp(-10**(0.4*d)),0)
    f['Schechter']=(sch,[0,mx.min()-4,-5],[hi*10,mx.min(),5])
    def cf(m,N,Ms,al,be):
        v=N*np.exp(al*(m-Ms))*(1-np.exp(be*(Ms-m))); return np.where(m>Ms,np.maximum(v,1e-4),1e-4)
    f['Ciardullo (free)']=(cf,[0,Msn,.05,1.0],[1e6,Msx,8,30])
    def cc2(m,N,Ms):
        v=N*np.exp(0.307*(m-Ms))*(1-np.exp(3.0*(Ms-m))); return np.where(m>Ms,np.maximum(v,1e-4),1e-4)
    f['Ciardullo (canonical)']=(cc2,[0,Msn],[1e6,Msx])
    return f

def run(sel,mask_bright,faint_cut):
    mx,my,ye,keep=build(sel,mask_bright,faint_cut)
    xf,yf_,ef=mx[keep],my[keep],ye[keep]
    out=[]
    for nm,(fn,lb,ub) in MODELS(xf,yf_,M[sel].min()).items():
        try:
            r=lambda p: np.sum(((yf_-fn(xf,*p))/ef)**2)
            de=differential_evolution(r,bounds=list(zip(lb,ub)),seed=42,maxiter=9000,tol=1e-12,popsize=35,mutation=(.3,1.9))
            po,_=curve_fit(fn,xf,yf_,p0=de.x,bounds=(lb,ub),sigma=ef,absolute_sigma=True,maxfev=200000)
            yy=fn(xf,*po); k=len(po); ch=np.sum(((yf_-yy)/ef)**2)
            r2=1-np.sum((yf_-yy)**2)/np.sum((yf_-yf_.mean())**2)
            out.append(dict(nm=nm,fn=fn,p=po,aic=ch+2*k,r2=r2,rc=ch/max(len(yf_)-k,1),k=k))
        except Exception: pass
    out.sort(key=lambda d:d['aic'])
    return mx,my,ye,keep,out

FIG=f'{B}/04_Figures'
summary={}
for tag,sel,mb,fc,title in SCEN:
    mx,my,ye,keep,res=run(sel,mb,fc)
    N=int(sel.sum()); title=title % N if '%d' in title else title
    summary[tag]=(N,int(keep.sum()),res)
    fig,axes=plt.subplots(5,2,figsize=(7.4,10.4),sharex=True,sharey=True)
    for i,r in enumerate(res):
        ax=axes.flat[i]
        ax.errorbar(mx[keep],my[keep],yerr=ye[keep],fmt='o',ms=3.1,color='k',lw=0,elinewidth=.8,capsize=1.5,zorder=5)
        if (~keep).sum():
            ax.errorbar(mx[~keep],my[~keep],yerr=ye[~keep],fmt='o',ms=3.1,mfc='white',mec='0.5',lw=0,elinewidth=.7,capsize=1.5,zorder=4)
        g=np.linspace(mx[keep].min()-0.35,mx[keep].max()+0.15,600)
        ax.plot(g,np.maximum(r['fn'](g,*r['p']),1e-3),lw=1.6,color='#c0392b' if i==0 else '#1f6fb4',zorder=6)
        if fc: ax.axvline(MLIM,ls=':',lw=.9,color='.5')
        if mb: ax.axvspan(ALO,AHI,color='k',alpha=.07)
        ax.set_yscale('log'); ax.set_ylim(.5,150); ax.set_xlim(-5.6,2.4)
        ax.set_title(f"{i+1}. {r['nm']}",fontsize=8.4,pad=2.5)
        ax.text(.03,.93,f"AIC {r['aic']:.1f}   $\\Delta$ {r['aic']-res[0]['aic']:.1f}   $R^2$ {r['r2']:.3f}",
                transform=ax.transAxes,va='top',fontsize=6.2,color='.25')
        ax.tick_params(labelsize=6.6)
        for s in ('top','right'): ax.spines[s].set_visible(False)
        if i%2==0: ax.set_ylabel(r'$N$ per 0.3 mag',fontsize=7.5)
        if i>=8:  ax.set_xlabel(r'$M_{\rm radio}$ / mag',fontsize=7.5)
    fig.suptitle("Radio PNLF — "+title,fontsize=10.5,y=0.995)
    fig.tight_layout(rect=[0,0,1,0.955])
    fig.savefig(f'{FIG}/step07g_pnlf_grid_A4_{tag}.pdf'); plt.close(fig)
    print(f"[{tag}] N={N} bins={int(keep.sum())}  best={res[0]['nm']}")

print("\nsummary by scenario")
for tag,(N,nb,res) in summary.items():
    print(f"\n### {tag}   N={N}  bins fitted={nb}")
    print(f"  {'rk':<4}{'model':<24}{'AIC':>8}{'dAIC':>8}{'R2':>8}{'chi2r':>8}   M*")
    for i,r in enumerate(res):
        ms=('%.3f'%r['p'][1]) if ('Ciardullo' in r['nm'] or r['nm']=='Schechter') else ''
        print(f"  {i+1:<4}{r['nm']:<24}{r['aic']:8.2f}{r['aic']-res[0]['aic']:8.2f}{r['r2']:8.3f}{r['rc']:8.3f}   {ms}")
