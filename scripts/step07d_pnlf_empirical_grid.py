"""Compact multi-panel versions of the Step 07a empirical PNLF fits.

The binning, bounds and random seed are the same as Step 07a, so the
panels reproduce those fits rather than refitting them differently.

Outputs
    04_Figures/step07d_pnlf_empirical_grid_10up.pdf   one 5x2 page, used in the paper
    04_Figures/step07d_pnlf_empirical_grid_4up.pdf    three 2x2 pages

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import os, warnings
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from scipy.optimize import differential_evolution, curve_fit
from scipy.stats import t as student_t, skewnorm
warnings.filterwarnings("ignore"); np.random.seed(42)

BASE=os.environ.get("PNBASE", os.path.expanduser("~/Desktop/Research/PN LMC Paper"))
T=Table.read(os.path.join(BASE,"03_Outputs","step07a_pnlf.vot"),format="votable")
L_REF=1e20; BIN_W=0.3; M_LIM=-0.5; M_ANOM_LO=-4.4; M_ANOM_HI=-3.7

log_lum=np.array(T["log_radio_lum"],dtype=float); valid=np.isfinite(log_lum)
M_all=-2.5*(log_lum[valid]-np.log10(L_REF))
edges=np.arange(np.floor(M_all.min()/BIN_W)*BIN_W,np.ceil(M_all.max()/BIN_W)*BIN_W+BIN_W,BIN_W)
cnts,_=np.histogram(M_all,bins=edges); ctrs=0.5*(edges[:-1]+edges[1:])
nz=cnts>0; mx=ctrs[nz]; my=cnts[nz].astype(float)
mye=np.sqrt(my); mye[mye==0]=1.0
anom=(mx>M_ANOM_LO)&(mx<M_ANOM_HI); incompl=mx>M_LIM; fit_m=~anom&~incompl
mx_f,my_f,mye_f=mx[fit_m],my[fit_m],mye[fit_m]
mx_o,my_o,mye_o=mx[~fit_m],my[~fit_m],mye[~fit_m]
m_peak=mx_f[np.argmax(my_f)]; Ms_max=M_all.min()-0.05; Ms_min=M_all.min()-5.0

def f_gaussian(m,A,mu,sig): return A*np.exp(-0.5*((m-mu)/sig)**2)
def f_studentt(m,A,mu,sig,nu): return A*student_t.pdf(m,max(nu,0.5),loc=mu,scale=max(abs(sig),0.01))
def f_cauchy(m,A,mu,g): return A/(1+((m-mu)/g)**2)
def f_skewnorm(m,A,loc,sc,ask): return A*skewnorm.pdf(m,ask,loc=loc,scale=sc)
def f_poly2(m,a,b,c): return a*m**2+b*m+c
def f_poly3(m,a,b,c,d): return a*m**3+b*m**2+c*m+d
def f_exponential(m,a,b,c): return a*np.exp(b*m)+c
def f_schechter(m,phi,ms,al):
    dm=m-ms; return np.maximum(phi*10**(0.4*(al+1)*dm)*np.exp(-10**(0.4*dm)),0)
def f_ciar_free(m,N0,Ms,al,be):
    v=N0*np.exp(al*(m-Ms))*(1-np.exp(be*(Ms-m))); return np.where(m>Ms,np.maximum(v,1e-4),1e-4)
def f_ciar_canonical(m,N0,Ms):
    v=N0*np.exp(0.307*(m-Ms))*(1-np.exp(3.0*(Ms-m))); return np.where(m>Ms,np.maximum(v,1e-4),1e-4)

defs=[("Gaussian",f_gaussian,[0,mx_f.min()-3,0.1],[my_f.max()*4,mx_f.max()+2,8.0]),
 ("Student-$t$",f_studentt,[0,mx_f.min()-3,0.1,0.5],[my_f.max()*3,mx_f.max()+2,10,200]),
 ("Cauchy (Lorentzian)",f_cauchy,[0,mx_f.min()-3,0.05],[my_f.max()*4,mx_f.max()+2,10.0]),
 ("Skew-normal",f_skewnorm,[0,mx_f.min()-4,0.1,-30],[my_f.max()*6,mx_f.max()+3,15,30]),
 ("Polynomial (2)",f_poly2,[-500,-200,0],[500,200,my_f.max()*3]),
 ("Polynomial (3)",f_poly3,[-500,-500,-200,0],[500,500,200,my_f.max()*3]),
 ("Exponential",f_exponential,[0,0,-50],[my_f.max()*10,5.0,my_f.max()]),
 ("Schechter",f_schechter,[0,mx_f.min()-4,-5],[my_f.max()*10,mx_f.min(),5]),
 ("Ciardullo (free)",f_ciar_free,[0,Ms_min,0.05,1.0],[1e6,Ms_max,8.0,30.0]),
 ("Ciardullo (canonical)",f_ciar_canonical,[0,Ms_min],[1e6,Ms_max])]

res={}
for nm,fn,lb,ub in defs:
    r=lambda p: np.sum(((my_f-fn(mx_f,*p))/mye_f)**2)
    de=differential_evolution(r,bounds=list(zip(lb,ub)),seed=42,maxiter=12000,tol=1e-13,popsize=40,mutation=(0.3,1.9))
    popt,_=curve_fit(fn,mx_f,my_f,p0=de.x,bounds=(lb,ub),sigma=mye_f,absolute_sigma=True,maxfev=300000)
    yf=fn(mx_f,*popt); k=len(popt)
    chi2=np.sum(((my_f-yf)/mye_f)**2)
    r2=1-np.sum((my_f-yf)**2)/np.sum((my_f-my_f.mean())**2)
    res[nm]=dict(fn=fn,p=popt,aic=chi2+2*k,r2=r2,redchi=chi2/(len(my_f)-k),k=k)
ranked=sorted(res.items(),key=lambda x:x[1]["aic"])

def panel(ax,nm,r,rank,small=False):
    ax.errorbar(mx_f,my_f,yerr=mye_f,fmt="o",ms=3.4,color="k",lw=0,elinewidth=0.8,capsize=1.6,zorder=5)
    ax.errorbar(mx_o,my_o,yerr=mye_o,fmt="o",ms=3.4,mfc="white",mec="0.45",lw=0,elinewidth=0.7,capsize=1.6,zorder=4)
    grid=np.linspace(mx_f.min()-0.35,M_LIM,600)
    ax.plot(grid,np.maximum(r["fn"](grid,*r["p"]),1e-3),lw=1.7,color="#c0392b" if rank==1 else "#1f6fb4",zorder=6)
    ax.axvline(M_LIM,ls=":",lw=1.0,color="0.5")
    ax.set_yscale("log"); ax.set_ylim(0.5,120); ax.set_xlim(-5.6,2.4)
    ax.set_title(f"{rank}. {nm}",fontsize=8.6 if small else 10,pad=3)
    ax.text(0.03,0.94,f"AIC {r['aic']:.1f}   $R^2$ {r['r2']:.3f}",transform=ax.transAxes,
            va="top",fontsize=6.6 if small else 8,color="0.25")
    ax.tick_params(labelsize=6.6 if small else 8.4)
    for s in ("top","right"): ax.spines[s].set_visible(False)

FIG=os.path.join(BASE,"04_Figures")
# --- 10-up single page ---
fig,axes=plt.subplots(2,5,figsize=(13.6,5.3),sharex=True,sharey=True)
for i,(nm,r) in enumerate(ranked):
    ax=axes.flat[i]; panel(ax,nm,r,i+1,small=True)
    if i%5==0: ax.set_ylabel(r"$N$ per 0.3 mag bin",fontsize=8)
    if i>=5: ax.set_xlabel(r"$M_{\rm radio}$ / mag",fontsize=8)
fig.suptitle("Radio PNLF — empirical model fits, ranked by AIC",fontsize=11.5,y=0.99)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(FIG,"step07d_pnlf_empirical_grid_10up.pdf")); plt.close(fig)

# --- 4-up, 3 pages ---
with PdfPages(os.path.join(FIG,"step07d_pnlf_empirical_grid_4up.pdf")) as pdf:
    for pg in range(3):
        fig,axes=plt.subplots(2,2,figsize=(7.6,6.4),sharex=True,sharey=True)
        for j in range(4):
            i=pg*4+j; ax=axes.flat[j]
            if i>=len(ranked): ax.axis("off"); continue
            nm,r=ranked[i]; panel(ax,nm,r,i+1)
            if j%2==0: ax.set_ylabel(r"$N$ per 0.3 mag bin",fontsize=9)
            if j>=2: ax.set_xlabel(r"$M_{\rm radio}$ / mag",fontsize=9)
        fig.suptitle(f"Radio PNLF — empirical model fits ({pg+1}/3)",fontsize=11)
        fig.tight_layout(rect=[0,0,1,0.95]); pdf.savefig(fig); plt.close(fig)
print("ranked:", [n for n,_ in ranked])
print("wrote both grid PDFs")
