"""
step07a_fit_pnlf_empirical.py — Step 07a: catalogue + model fits
================================================================
Each model page:  fit curve | residuals+stats | general eq (blue) | fitted eq (yellow)

The primary PNLF remains a MeerKAT 1.295-GHz luminosity function.  ASKAP-only
sources are retained in the input/master catalogue but are explicitly excluded
from the PNLF fit, because combining 888-MHz and 1.295-GHz fluxes would mix
frequencies, angular resolutions, sensitivities and completeness functions.

Models fitted (9 total):
  Empirical symmetric  : Gaussian, Student-t, Cauchy (Lorentzian)
  Empirical asymmetric : Skew-normal
  Polynomial           : Polynomial (2), Polynomial (3)
  Exponential          : Exponential
  Astrophysical        : Schechter, Ciardullo (free alpha, beta, M*, N0)

Note on Ciardullo: alpha and beta are FREE (not fixed to optical canonical values).
This gives the most general radio PNLF fit. The step7b script shows the comparison
between canonical (fixed) and free versions.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _support.plot_style import apply_paper_style
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
from astropy.table import Table, Column
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import skewnorm, t as student_t

warnings.filterwarnings("ignore")
apply_paper_style()
np.random.seed(42)

BASE=os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUTS=os.path.join(BASE,"03_Outputs"); FIGS=os.path.join(BASE,"04_Figures")
INPUT_FILE=os.path.join(OUTS,"step06b_confidence.vot")
OUT_VOT=os.path.join(OUTS,"step07a_pnlf.vot")
OUT_FIG=os.path.join(FIGS,"step07a_pnlf_empirical_fits.pdf")

# D_KPC=49.97; D_CM=D_KPC*3.0856776e21; L_REF=1e20
# updated 2026-08-30: Pietrzynski et al. 2019 gives 49.59 kpc (mu = 18.477); 49.97 kpc was inconsistent with the quoted modulus
D_KPC=49.59; D_CM=D_KPC*3.0856776e21; L_REF=1e20
BIN_W=0.3; M_LIM=-0.5; M_ANOM_LO=-4.4; M_ANOM_HI=-3.7
# S_MLIM_UJY=53.05; RMS_MEERKAT=10.0
# updated 2026-08-30: median rms 11 uJy/beam from Cotton et al. 2026 / Rajabpour et al. 2026; S(M_lim) recomputed at d = 49.59 kpc
S_MLIM_UJY=53.87; RMS_MEERKAT=11.0

print("step07a: fitting the empirical radio PNLF models")

print("\n[1] Loading & computing luminosity ...")
det=Table.read(INPUT_FILE,format="votable")
# The PNLF is frequency-specific: always prefer the explicitly named MeerKAT
# 1.295-GHz broadband flux. The compatibility alias is only a fallback.
if   "mkt_int_flux_Jy" in det.colnames: rfj=np.array(det["mkt_int_flux_Jy"],dtype=float)
elif "radio_flux_Jy"   in det.colnames: rfj=np.array(det["radio_flux_Jy"],  dtype=float)
elif "ch0_int_flux"    in det.colnames: rfj=np.array(det["ch0_int_flux"],   dtype=float)
else: raise ValueError("No radio flux column.")
has_meerkat=(np.array(det["det_meerkat"],dtype=int)==1)
valid=has_meerkat&np.isfinite(rfj)&(rfj>0)
lum=np.full(len(det),np.nan); log_lum=np.full(len(det),np.nan)
lum[valid]=4*np.pi*D_CM**2*rfj[valid]*1e-23; log_lum[valid]=np.log10(lum[valid])
n_valid=int(valid.sum())
print(f"    {n_valid} sources  |  log L: {np.nanmin(log_lum):.3f}–{np.nanmax(log_lum):.3f}")
for col in ["radio_lum_cgs","log_radio_lum","pnlf_included","pnlf_exclusion_reason"]:
    if col in det.colnames: det.remove_column(col)
det.add_column(Column(lum.astype("f8"),    name="radio_lum_cgs",  description="Radio luminosity [erg/s/Hz]"))
det.add_column(Column(log_lum.astype("f4"),name="log_radio_lum",  description="log10 radio luminosity"))
det.add_column(Column(valid.astype("i2"), name="pnlf_included",
                      description="1 only for valid MeerKAT 1.295-GHz PNLF input"))
pnlf_reason=np.full(len(det),"included_MeerKAT_1295",dtype="U40")
pnlf_reason[~has_meerkat]="excluded_ASKAP_only_frequency_mismatch"
pnlf_reason[has_meerkat&~(np.isfinite(rfj)&(rfj>0))]="excluded_invalid_MeerKAT_flux"
det.add_column(Column(pnlf_reason,name="pnlf_exclusion_reason"))
M_col=np.full(len(det),np.nan); M_col[valid]=-2.5*(log_lum[valid]-np.log10(L_REF))
for col in ["M_radio"]:
    if col in det.colnames: det.remove_column(col)
det.add_column(Column(M_col.astype("f4"),name="M_radio",description="Radio absolute magnitude: -2.5*log10(L/1e20)"))
det.write(OUT_VOT,format="votable",overwrite=True)
print(f"    -> {OUT_VOT}")

M_all=-2.5*(log_lum[valid]-np.log10(L_REF))
edges=np.arange(np.floor(M_all.min()/BIN_W)*BIN_W,np.ceil(M_all.max()/BIN_W)*BIN_W+BIN_W,BIN_W)
cnts,_=np.histogram(M_all,bins=edges); ctrs=0.5*(edges[:-1]+edges[1:])
nz=cnts>0; mx=ctrs[nz]; my=cnts[nz].astype(float)
mye=np.sqrt(my); mye[mye==0]=1.0
anom=(mx>M_ANOM_LO)&(mx<M_ANOM_HI); incompl=mx>M_LIM; fit_m=~anom&~incompl
mx_f=mx[fit_m]; my_f=my[fit_m]; mye_f=mye[fit_m]
mx_o=mx[~fit_m]; my_o=my[~fit_m]; mye_o=mye[~fit_m]
m_peak=mx_f[np.argmax(my_f)]; Ms_max=M_all.min()-0.05; Ms_min=M_all.min()-5.0
print(f"\n[2] Bins in fit: {fit_m.sum()}  excluded: {(~fit_m).sum()}"
      f"  (anom={anom.sum()}, incompl={incompl.sum()})")

# Symmetric distributions
def f_gaussian(m,A,mu,sig):
    return A*np.exp(-0.5*((m-mu)/sig)**2)

def f_studentt(m,A,mu,sig,nu):
    sig=max(abs(sig),0.01); nu=max(nu,0.5)
    return A*student_t.pdf(m,nu,loc=mu,scale=sig)

def f_cauchy(m,A,mu,gamma):
    """Cauchy / Lorentzian: heavy-tailed symmetric distribution."""
    return A/(1+((m-mu)/gamma)**2)

# Asymmetric
def f_skewnorm(m,A,loc,sc,ask):
    return A*skewnorm.pdf(m,ask,loc=loc,scale=sc)

# Polynomial
def f_poly2(m,a,b,c):  return a*m**2+b*m+c
def f_poly3(m,a,b,c,d): return a*m**3+b*m**2+c*m+d

# Exponential
def f_exponential(m,a,b,c):
    """Simple exponential: N(M) = a*exp(b*M) + c"""
    return a*np.exp(b*m)+c

# Astrophysical
def f_schechter(m,phi,ms,alpha):
    dm=m-ms; return np.maximum(phi*10**(0.4*(alpha+1)*dm)*np.exp(-10**(0.4*dm)),0)

def f_ciar_free(m,N0,Ms,alpha,beta):
    """
    Ciardullo (1989) PNLF with ALL parameters free:
      N(M) = N0 * exp(alpha*(M-M*)) * (1 - exp(beta*(M*-M)))
    alpha >= 0.05 (faint-end slope, free)
    beta  >= 1.0  (bright-end cutoff sharpness, free)
    """
    v=N0*np.exp(alpha*(m-Ms))*(1-np.exp(beta*(Ms-m)))
    return np.where(m>Ms,np.maximum(v,1e-4),1e-4)

def f_ciar_canonical(m,N0,Ms):
    """
    Ciardullo (1989) PNLF with canonical optical parameters fixed:
      alpha = 0.307 (faint-end slope, fixed)
      beta  = 3.0   (bright-end cutoff, fixed)
    Only N0 (amplitude) and M* (bright cutoff) are free — k=2.
    """
    v=N0*np.exp(0.307*(m-Ms))*(1-np.exp(3.0*(Ms-m)))
    return np.where(m>Ms,np.maximum(v,1e-4),1e-4)

# Equation strings shown on the figure panels
def pm(v): return "+ " if v>=0 else "- "

def make_equations(nm, popt):
    """Plain Unicode strings. No matplotlib mathtext — zero parse errors."""
    p=popt
    if nm=="Gaussian":
        gen=("N(M) = A * exp[ -0.5 * ((M - mu) / sigma)^2 ]\n"
             "A = amplitude,  mu = peak magnitude,  sigma = width")
        fit="N(M) = {:.2f} * exp[ -0.5 * ((M - ({:.3f})) / {:.3f})^2 ]".format(p[0],p[1],p[2])

    elif nm=="Student-t":
        gen=("N(M) = A * t_nu( (M - mu) / sigma )\n"
             "t_nu(x) proportional to (1 + x^2/nu)^(-(nu+1)/2)\n"
             "A = amplitude,  mu = centre,  sigma = scale,  nu = degrees of freedom")
        fit="N(M) = {:.2f} * t_{:.1f}( (M - ({:.3f})) / {:.3f} )".format(p[0],p[3],p[1],p[2])

    elif nm=="Cauchy (Lorentzian)":
        gen=("N(M) = A / [ 1 + ((M - mu) / gamma)^2 ]\n"
             "Heavy-tailed symmetric distribution (heavier tails than Gaussian)\n"
             "A = peak height,  mu = centre,  gamma = half-width at half-maximum")
        fit="N(M) = {:.2f} / [ 1 + ((M - ({:.3f})) / {:.3f})^2 ]".format(p[0],p[1],p[2])

    elif nm=="Skew-normal":
        gen=("N(M) = A * f_SN(M; mu, sigma, alpha_sk)\n"
             "f_SN(x) = (2/sigma) * phi((x-mu)/sigma) * Phi(alpha_sk*(x-mu)/sigma)\n"
             "phi = normal PDF,  Phi = normal CDF,  alpha_sk < 0 = left-skewed")
        fit="N(M) = {:.2f} * f_SN(M;  mu={:.3f},  sigma={:.3f},  alpha_sk={:.3f})".format(p[0],p[1],p[2],p[3])

    elif nm=="Polynomial (2)":
        gen="N(M) = a*M^2 + b*M + c"
        fit="N(M) = {:.4f}*M^2  {}  {:.4f}*M  {}  {:.4f}".format(p[0],pm(p[1]),abs(p[1]),pm(p[2]),abs(p[2]))

    elif nm=="Polynomial (3)":
        gen="N(M) = a*M^3 + b*M^2 + c*M + d"
        fit=("N(M) = {:.4f}*M^3  {}  {:.4f}*M^2  {}  {:.4f}*M  {}  {:.4f}"
             .format(p[0],pm(p[1]),abs(p[1]),pm(p[2]),abs(p[2]),pm(p[3]),abs(p[3])))

    elif nm=="Exponential":
        gen=("N(M) = a * exp(b * M) + c\n"
             "Simple exponential rise toward faint magnitudes\n"
             "a = amplitude,  b = rate (>0 = rising toward faint),  c = offset")
        fit="N(M) = {:.4f} * exp({:.4f} * M)  {}  {:.4f}".format(p[0],p[1],pm(p[2]),abs(p[2]))

    elif nm=="Schechter":
        gen=("N(M) = phi * 10^(0.4*(alpha+1)*(M-M*)) * exp[-10^(0.4*(M-M*))]\n"
             "phi = normalisation,  M* = bright cutoff,  alpha = faint-end slope\n"
             "Originally for galaxy luminosity functions (Schechter 1976)")
        fit=("N(M) = {:.3f} * 10^(0.4*({:.3f}+1)*(M-({:.3f})))\n"
             "       * exp[-10^(0.4*(M-({:.3f})))]").format(p[0],p[2],p[1],p[1])

    elif nm=="Ciardullo (free)":
        Lstar=L_REF*10**(-p[1]/2.5)
        gen=("N(M) = N0 * exp(alpha*(M - M*)) * (1 - exp(beta*(M* - M)))\n"
             "ALL parameters free: N0, M*, alpha (faint-end slope), beta (bright cutoff)\n"
             "Ciardullo (1989) form — radio calibration, not fixed to optical values")
        fit=("N(M) = {:.2f} * exp({:.3f}*(M - ({:.3f}))) * (1 - exp({:.3f}*({:.3f} - M)))\n"
             "N0={:.2f}  M*={:.3f}  alpha={:.3f}  beta={:.3f}\n"
             "L* = 10^{:.3f} erg/s/Hz"
             .format(p[0],p[2],p[1],p[3],p[1],p[0],p[1],p[2],p[3],np.log10(Lstar)))

    elif nm=="Ciardullo (canonical)":
        Lstar=L_REF*10**(-p[1]/2.5)
        gen=("N(M) = N0 * exp(0.307*(M - M*)) * (1 - exp(3.0*(M* - M)))\n"
             "alpha = 0.307,  beta = 3.0  (FIXED — Ciardullo 1989 canonical optical values)\n"
             "Only N0 and M* are free  (k=2) — most parsimonious Ciardullo form")
        fit=("N(M) = {:.2f} * exp(0.307*(M - ({:.3f}))) * (1 - exp(3.0*({:.3f} - M)))\n"
             "N0={:.2f}  M*={:.3f}  alpha=0.307 (fixed)  beta=3.0 (fixed)\n"
             "L* = 10^{:.3f} erg/s/Hz"
             .format(p[0],p[1],p[1],p[0],p[1],np.log10(Lstar)))
    else:
        gen=f"Model: {nm}"; fit=str(popt)
    return gen, fit

def cstats(y,yf,ye,k):
    ss_r=np.sum((y-yf)**2); ss_t=np.sum((y-y.mean())**2)
    r2=1-ss_r/ss_t if ss_t>0 else np.nan
    chi2=np.sum(((y-yf)/ye)**2); dof=len(y)-k
    return dict(r2=r2,chi2=chi2,redchi=chi2/dof if dof>0 else np.nan,
                aic=chi2+2*k,bic=chi2+k*np.log(len(y)),k=k,dof=dof)

def try_fit(nm,func,lb,ub,p0):
    def res(p): return np.sum(((my_f-func(mx_f,*p))/mye_f)**2)
    try:
        de=differential_evolution(res,bounds=list(zip(lb,ub)),seed=42,maxiter=12000,
                                   tol=1e-13,popsize=40,mutation=(0.3,1.9))
        popt,pcov=curve_fit(func,mx_f,my_f,p0=de.x,bounds=(lb,ub),
                             sigma=mye_f,absolute_sigma=True,maxfev=300000)
        yfit=func(mx_f,*popt); perr=np.sqrt(np.diag(pcov))
        s=cstats(my_f,yfit,mye_f,len(popt))
        return dict(popt=popt,perr=perr,pcov=pcov,lb=np.array(lb,dtype=float),
                    ub=np.array(ub,dtype=float),func=func,s=s,yfit=yfit)
    except Exception as e:
        print(f"    {nm} FAILED: {e}"); return None

print("\n[3] Fitting all models ...")
results={}

# Model definitions: (name, function, lower_bounds, upper_bounds, initial_guess)
model_defs = [
    # Symmetric
    ("Gaussian",
     f_gaussian,
     [0, mx_f.min()-3, 0.1],
     [my_f.max()*4, mx_f.max()+2, 8.0],
     [my_f.max(), m_peak, 1.5]),

    ("Student-t",
     f_studentt,
     [0, mx_f.min()-3, 0.1, 0.5],
     [my_f.max()*3, mx_f.max()+2, 10, 200],
     [my_f.max()*2, m_peak, 1.5, 5]),

    ("Cauchy (Lorentzian)",
     f_cauchy,
     [0, mx_f.min()-3, 0.05],
     [my_f.max()*4, mx_f.max()+2, 10.0],
     [my_f.max(), m_peak, 1.0]),

    # Asymmetric
    ("Skew-normal",
     f_skewnorm,
     [0, mx_f.min()-4, 0.1, -30],
     [my_f.max()*6, mx_f.max()+3, 15, 30],
     [my_f.max()*3, m_peak, 2, -2]),

    # Polynomial
    ("Polynomial (2)",
     f_poly2,
     [-500, -200, 0],
     [500, 200, my_f.max()*3],
     [-50, 0, my_f.max()]),

    ("Polynomial (3)",
     f_poly3,
     [-500, -500, -200, 0],
     [500, 500, 200, my_f.max()*3],
     [0, -50, 0, my_f.max()]),

    # Exponential
    ("Exponential",
     f_exponential,
     [0, 0, -50],
     [my_f.max()*10, 5.0, my_f.max()],
     [1.0, 0.5, 0]),

    # Astrophysical
    ("Schechter",
     f_schechter,
     [0, mx_f.min()-4, -5],
     [my_f.max()*10, mx_f.min(), 5],
     [my_f.max()*3, mx_f.min()-0.5, -1]),

    ("Ciardullo (free)",
     f_ciar_free,
     [0, Ms_min, 0.05, 1.0],      # N0, M*, alpha>=0.05, beta>=1.0
     [1e6, Ms_max, 8.0, 30.0],
     [my_f.max()*3, Ms_max-0.5, 0.307, 3.0]),

    ("Ciardullo (canonical)",
     f_ciar_canonical,
     [0, Ms_min],                  # N0, M* only (alpha=0.307, beta=3.0 fixed)
     [1e6, Ms_max],
     [my_f.max()*3, Ms_max-0.5]),
]

for nm,func,lb,ub,p0 in model_defs:
    r=try_fit(nm,func,lb,ub,p0)
    if r:
        results[nm]=r
        print(f"    {nm:<25}  R²={r['s']['r2']:.4f}  AIC={r['s']['aic']:.2f}  "
              f"red-χ²={r['s']['redchi']:.3f}  k={r['s']['k']}")

ranked=sorted(results.items(),key=lambda x:x[1]["s"]["aic"])
best_name,best_r=ranked[0]

print(f"\n[4] Ranking by AIC:")
print(f"    {'Rk':<3} {'Model':<25} {'R²':>7} {'AIC':>8} {'BIC':>8} {'red-χ²':>8} {'k':>3} {'DOF':>4}")
print(f"    {'-'*70}")
for i,(nm,r) in enumerate(ranked):
    tag=" <- BEST" if i==0 else ""
    print(f"    {i+1:<3d} {nm:<25} {r['s']['r2']:>7.4f} {r['s']['aic']:>8.2f} "
          f"{r['s']['bic']:>8.2f} {r['s']['redchi']:>8.3f} {r['s']['k']:>3d} {r['s']['dof']:>4d}{tag}")

print(f"\n[5] Generating PDF ...")

def sigma_band(func, popt, pcov, x_arr):
    """Jacobian 1-sigma band, sigma_y(x) = sqrt(J^T . pcov . J).

    Linear error propagation, so the band is symmetric about the best fit.
    """
    popt=np.asarray(popt,dtype=float); pcov=np.asarray(pcov,dtype=float)
    x_arr=np.asarray(x_arr,dtype=float); n_p=len(popt); n_x=len(x_arr)
    J=np.zeros((n_x,n_p))
    for j in range(n_p):
        h=max(abs(popt[j])*1e-5,1e-8)
        pp=popt.copy(); pp[j]+=h
        pm=popt.copy(); pm[j]-=h
        try:
            J[:,j]=(func(x_arr,*pp)-func(x_arr,*pm))/(2*h)
        except Exception:
            J[:,j]=0.0
    var=np.einsum('ij,jk,ik->i',J,pcov,J)
    return np.sqrt(np.maximum(var,0))

COLS={
    "Gaussian"            :"#1976D2",
    "Student-t"           :"#9C27B0",
    "Cauchy (Lorentzian)" :"#E91E63",
    "Skew-normal"         :"#4CAF50",
    "Polynomial (2)"      :"#D32F2F",
    "Polynomial (3)"      :"#FF9800",
    "Exponential"         :"#795548",
    "Schechter"           :"#00BCD4",
    "Ciardullo (free)"    :"#1A1A1A",
    "Ciardullo (canonical)":"#B71C1C",  # dark red — physical model, canonical form
}
def gcol(nm,ci): return COLS.get(nm,["#607D8B","#F57F17"][ci%2])
m_fine=np.linspace(M_all.min()-0.5,M_all.max()+0.5,600)

def plot_data(ax):
    ax.errorbar(mx_f,my_f,yerr=mye_f,fmt="ko",capsize=4,markersize=8,elinewidth=1.3,
                capthick=1.3,zorder=20,label=f"Data  (M <= {M_LIM}, complete)")
    ax.errorbar(mx_o,my_o,yerr=mye_o,fmt="o",capsize=4,markersize=8,mfc="white",
                mec="black",ecolor="black",elinewidth=1.0,capthick=1.0,
                linestyle="none",zorder=20,label=f"Excluded  (M > {M_LIM} or anomalous)")

def fmt_ax(ax):
    ax.axvline(M_LIM,color="#aaa",ls="--",lw=1.2,zorder=1)
    ax.text(M_LIM+0.07,0.95,f"Mlim={M_LIM}\n(5sig~{S_MLIM_UJY:.0f} uJy)",
            fontsize=8.5,color="#666",ha="left",va="top",transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.8,ec="none"))
    ax.set_xlabel(r"$M_{\rm radio} = -2.5\,\log_{10}(L/10^{20})$  [erg s$^{-1}$ Hz$^{-1}$]",fontsize=12)
    ax.set_ylabel(r"$N$ (per bin)",fontsize=12)
    ax.set_yscale("log"); ax.set_xlim(M_all.min()-0.4,M_all.max()+0.4)
    ax.set_ylim(0.4,my.max()*6); ax.tick_params(labelsize=10)
    ax.spines[["top","right"]].set_visible(False)

with PdfPages(OUT_FIG) as pdf:

    # Page 1: Data preparation pipeline (4 panels)
    fig_dp = plt.figure(figsize=(14, 10))
    fig_dp.patch.set_facecolor("white")
    fig_dp.suptitle("Data Preparation: From Luminosity to Fit-Ready PNLF",
                     fontsize=14, fontweight="bold", y=0.97)
    gs_dp = GridSpec(2, 2, figure=fig_dp, hspace=0.35, wspace=0.30)

    L_valid = lum[valid]  # linear luminosity array

    # (a) Histogram of L (luminosity in linear units)
    ax_a = fig_dp.add_subplot(gs_dp[0, 0])
    ax_a.hist(L_valid, bins=30, color="#1976D2", alpha=0.75, edgecolor="white", lw=0.8)
    ax_a.set_xlabel(r"$L_{\rm 1.3\,GHz}$  [erg s$^{-1}$ Hz$^{-1}$]", fontsize=11)
    ax_a.set_ylabel("Number of PNe", fontsize=11)
    ax_a.set_title("(a)  Luminosity distribution", fontsize=12, fontweight="bold")
    ax_a.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
    ax_a.spines[["top","right"]].set_visible(False)
    ax_a.tick_params(labelsize=9)
    ax_a.text(0.95, 0.95, f"N = {n_valid}\nD = {D_KPC} kpc",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"))

    # (b) Histogram of M (magnitude)
    ax_b = fig_dp.add_subplot(gs_dp[0, 1])
    ax_b.hist(M_all, bins=edges, color="#4CAF50", alpha=0.75, edgecolor="white", lw=0.8)
    ax_b.set_xlabel(r"$M_{\rm radio} = -2.5\,\log_{10}(L/10^{20})$  [mag]", fontsize=11)
    ax_b.set_ylabel("Number of PNe", fontsize=11)
    ax_b.set_title("(b)  Magnitude distribution", fontsize=12, fontweight="bold")
    ax_b.axvline(M_LIM, color="#aaa", ls="--", lw=1.2)
    ax_b.text(M_LIM + 0.1, ax_b.get_ylim()[1]*0.9, f"M_lim={M_LIM}",
              fontsize=8.5, color="#666", ha="left", va="top")
    ax_b.spines[["top","right"]].set_visible(False)
    ax_b.tick_params(labelsize=9)
    ax_b.text(0.95, 0.95, f"bin = {BIN_W} mag\n{len(edges)-1} bins total",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"))

    # (c) Histogram bars + error bars overlaid
    ax_c = fig_dp.add_subplot(gs_dp[1, 0])
    ax_c.bar(mx, my, width=BIN_W*0.85, color="#FF9800", alpha=0.55,
             edgecolor="#E65100", lw=0.8, zorder=5, label="Histogram counts")
    ax_c.errorbar(mx, my, yerr=mye, fmt="ko", capsize=4, markersize=5,
                  elinewidth=1.2, capthick=1.2, zorder=10,
                  label=r"$\sqrt{N}$ Poisson errors")
    ax_c.set_xlabel(r"$M_{\rm radio}$  [mag]", fontsize=11)
    ax_c.set_ylabel(r"$N$ (per bin)", fontsize=11)
    ax_c.set_title("(c)  Binned counts with Poisson uncertainties",
                    fontsize=12, fontweight="bold")
    ax_c.spines[["top","right"]].set_visible(False)
    ax_c.tick_params(labelsize=9)
    ax_c.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax_c.axvline(M_LIM, color="#aaa", ls="--", lw=1.2)

    # (d) Fit-ready data points: complete vs excluded
    ax_d = fig_dp.add_subplot(gs_dp[1, 1])
    ax_d.errorbar(mx_f, my_f, yerr=mye_f, fmt="ko", capsize=4, markersize=8,
                  elinewidth=1.3, capthick=1.3, zorder=20,
                  label=f"Fit bins  (M <= {M_LIM})")
    ax_d.errorbar(mx_o, my_o, yerr=mye_o, fmt="o", capsize=4, markersize=8,
                  mfc="white", mec="black", ecolor="black",
                  elinewidth=1.0, capthick=1.0, linestyle="none", zorder=20,
                  label=f"Excluded  (M > {M_LIM} or anomalous)")
    ax_d.axvline(M_LIM, color="#aaa", ls="--", lw=1.2)
    ax_d.text(M_LIM + 0.07, 0.95, f"Mlim={M_LIM}\n(5sig~{S_MLIM_UJY:.0f} uJy)",
              fontsize=8.5, color="#666", ha="left", va="top",
              transform=ax_d.get_xaxis_transform(),
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))
    # shade anomalous region
    ax_d.axvspan(M_ANOM_LO, M_ANOM_HI, color="#FFCDD2", alpha=0.3, zorder=1,
                 label=f"Anomalous zone ({M_ANOM_LO} to {M_ANOM_HI})")
    ax_d.set_xlabel(r"$M_{\rm radio}$  [mag]", fontsize=11)
    ax_d.set_ylabel(r"$N$ (per bin)", fontsize=11)
    ax_d.set_title("(d)  Fit-ready data: complete vs excluded",
                    fontsize=12, fontweight="bold")
    ax_d.set_yscale("log")
    ax_d.set_ylim(0.4, my.max()*6)
    ax_d.spines[["top","right"]].set_visible(False)
    ax_d.tick_params(labelsize=9)
    ax_d.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax_d.text(0.97, 0.03,
              f"Fit: {fit_m.sum()} bins  |  Excl: {(~fit_m).sum()} bins\n"
              f"(anom={anom.sum()}, incompl={incompl.sum()})",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig_dp, dpi=150)
    plt.close(fig_dp)

    # Page 2: all overlaid
    fig,ax=plt.subplots(figsize=(12,7)); fig.patch.set_facecolor("white")
    plot_data(ax)
    for ci,(nm,r) in enumerate(ranked):
        col=gcol(nm,ci); lw=2.8 if ci==0 else 1.5   # rank-#1 gets thick line
        yf=r["func"](m_fine,*r["popt"])
        ax.plot(m_fine,np.where(yf>0.3,yf,np.nan),"-",color=col,lw=lw,alpha=0.9,
                label=f"{nm}  (R²={r['s']['r2']:.3f}, AIC={r['s']['aic']:.1f})")
    fmt_ax(ax)
    ax.legend(fontsize=8,framealpha=0.95,loc="upper left",handlelength=2.0,labelspacing=0.35,ncol=2)
    ax.text(0.98,0.03,f"N={n_valid}  bin={BIN_W} mag  D={D_KPC} kpc",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.9))
    plt.tight_layout(); pdf.savefig(fig,dpi=150); plt.close(fig)

    # Pages 2+: one model per page
    for ci,(nm,r) in enumerate(ranked):
        col=gcol(nm,ci)
        gen_eq,fit_eq=make_equations(nm,r["popt"])

        fig=plt.figure(figsize=(14,8.5)); fig.patch.set_facecolor("white")
        gs=GridSpec(2,2,figure=fig,height_ratios=[3.5,1.0],width_ratios=[2,1],hspace=0.08,wspace=0.3)
        ax=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax_eq=fig.add_subplot(gs[1,:])

        # Only complete bins were fitted, so the curve stops at M_lim.
        m_plot = m_fine[m_fine <= M_LIM]   # solid curve stops at M_lim
        # 1-sigma band via Jacobian propagation (symmetric around best fit)
        try:
            yfit_plot=r["func"](m_plot,*r["popt"])
            sig=sigma_band(r["func"],r["popt"],r["pcov"],m_plot)
            lo=yfit_plot-sig; hi=yfit_plot+sig
            ymax_vis=my.max()*6
            ax.fill_between(m_plot,np.clip(lo,0.3,ymax_vis),np.clip(hi,0.3,ymax_vis),
                            color=col,alpha=0.15)
        except Exception as e:
            print(f"      Band failed for {nm}: {e}")
        yf=r["func"](m_plot,*r["popt"])
        ax.plot(m_plot,np.where(yf>0.3,yf,np.nan),"-",color=col,lw=2.5,label=f"Rank #{ci+1}: {nm}")
        plot_data(ax); fmt_ax(ax)
        ax.legend(fontsize=10,framealpha=0.95,loc="upper left")
        ax.set_title(f"Rank #{ci+1}: {nm}",fontsize=13,fontweight="bold")

        # Residuals
        res=my_f-r["func"](mx_f,*r["popt"])
        ax2.bar(mx_f,res,width=BIN_W*0.8,color=[col if v>=0 else "#555" for v in res],alpha=0.75,zorder=5)
        ax2.axhline(0,color="black",lw=1.0)
        ax2.errorbar(mx_f,res,yerr=mye_f,fmt="none",ecolor="black",elinewidth=1.0,capsize=3)
        ax2.set_xlabel("M_radio  [mag]",fontsize=11); ax2.set_ylabel("Residuals (data-model)",fontsize=10)
        ax2.set_title("Residuals",fontsize=11); ax2.spines[["top","right"]].set_visible(False)

        # Stats box
        s=r["s"]; rms=np.sqrt(np.mean(res**2)); p,pe=r["popt"],r["perr"]
        # Per-model parameter names (generic p0/p1/... if model not in dict)
        PARAM_NAMES={
            "Gaussian"             :["A","mu","sigma"],
            "Student-t"            :["A","mu","sigma","nu"],
            "Cauchy (Lorentzian)"  :["A","mu","gamma"],
            "Skew-normal"          :["A","mu","sigma","alpha_sk"],
            "Polynomial (2)"       :["a","b","c"],
            "Polynomial (3)"       :["a","b","c","d"],
            "Exponential"          :["a","b","c"],
            "Schechter"            :["phi","M*","alpha"],
            "Ciardullo (free)"     :["N0","M*","alpha","beta"],
            "Ciardullo (canonical)":["N0","M*"],
        }
        names=PARAM_NAMES.get(nm,[f"p{j}" for j in range(len(p))])
        wname=max(len(n) for n in names)
        plines=["{:<{w}} = {:.4f} +/- {:.4f}".format(n,v,e,w=wname)
                for n,v,e in zip(names,p,pe)]
        st="\n".join(["Model: {}".format(nm),"Rank: #{} (by AIC)".format(ci+1),"─"*32,
            "AIC    = {:.2f}".format(s["aic"]),"BIC    = {:.2f}".format(s["bic"]),
            "chi2   = {:.2f}".format(s["chi2"]),"red-x2 = {:.3f}".format(s["redchi"]),
            "R2     = {:.4f}".format(s["r2"]),"params = {}".format(s["k"]),
            "DOF    = {}".format(s["dof"]),"RMS    = {:.3f}".format(rms),"─"*32,
            ]+plines+["─"*32,"N_fit={} bins  N_src={}".format(len(mx_f),n_valid),
                      "D={} kpc  BIN_W={} mag".format(D_KPC,BIN_W)])
        at=AnchoredText(st,prop=dict(size=8,family="monospace"),frameon=True,
                        loc="lower left",bbox_to_anchor=(0.01,0.01),bbox_transform=ax2.transAxes)
        at.patch.set_boxstyle("round,pad=0.4"); at.patch.set_alpha(0.95); ax2.add_artist(at)

        # Equation panel — side-by-side, compact font, lowered to clear x-axis
        ax_eq.axis("off")
        ax_eq.plot([0.01,0.99],[0.99,0.99],color="#ccc",lw=0.8,transform=ax_eq.transAxes,clip_on=False)
        ax_eq.text(0.02,0.62,"General model equation:",transform=ax_eq.transAxes,
                   fontsize=8,fontweight="bold",color="#333",va="top",ha="left")
        ax_eq.text(0.02,0.38,gen_eq,transform=ax_eq.transAxes,fontsize=7.5,
                   va="top",ha="left",family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4",fc="#EEF5FF",alpha=0.95,ec="#90CAF9",lw=1.2))
        ax_eq.text(0.52,0.62,"Fitted equation (parameter values substituted):",
                   transform=ax_eq.transAxes,fontsize=8,fontweight="bold",color="#333",va="top",ha="left")
        ax_eq.text(0.52,0.38,fit_eq,transform=ax_eq.transAxes,fontsize=7.5,
                   va="top",ha="left",family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4",fc="#FFF8E1",alpha=0.95,ec="#FFD54F",lw=1.2))

        plt.tight_layout(rect=[0,0,1,1])
        pdf.savefig(fig,dpi=150,bbox_inches="tight"); plt.close(fig)

print(f"    -> {OUT_FIG}  ({len(ranked)+2} pages)")

print("\nstep07a complete: model comparison summary")
print(f"  N={n_valid}  M: {M_all.min():.2f}–{M_all.max():.2f}  M_lim={M_LIM}")
print(f"  {len(results)} models fitted  |  {len(mx_f)} bins used  |  {len(mx_o)} bins excluded")
print(f"\n  {'Rk':<3} {'Model':<25} {'R²':>7} {'AIC':>8} {'BIC':>8} {'red-χ²':>7} {'k':>3}")
print(f"  {'-'*65}")
for i,(nm,r) in enumerate(ranked):
    tag=" <- BEST" if i==0 else ""
    print(f"  {i+1:<3} {nm:<25} {r['s']['r2']:>7.4f} {r['s']['aic']:>8.2f} "
          f"{r['s']['bic']:>8.2f} {r['s']['redchi']:>7.3f} {r['s']['k']:>3d}{tag}")

p,pe=best_r["popt"],best_r["perr"]
print(f"\n  BEST: {best_name}")
print(f"    AIC={best_r['s']['aic']:.2f}  R²={best_r['s']['r2']:.4f}  red-χ²={best_r['s']['redchi']:.3f}")
_,fit_best=make_equations(best_name,p)
print(f"    Fitted: {fit_best.split(chr(10))[0]}")

if "Ciardullo (free)" in results:
    rc=results["Ciardullo (free)"]; pp=rc["popt"]
    Ls=L_REF*10**(-pp[1]/2.5)
    rank_c=[nm for nm,_ in ranked].index("Ciardullo (free)")+1
    print(f"\n  Ciardullo (free) — rank #{rank_c}:")
    print(f"    N(M) = {pp[0]:.2f} * exp({pp[2]:.3f}*(M-({pp[1]:.4f}))) * (1-exp({pp[3]:.3f}*({pp[1]:.4f}-M)))")
    print(f"    N0={pp[0]:.2f}  M*={pp[1]:.4f}  alpha={pp[2]:.3f}  beta={pp[3]:.3f}")
    print(f"    L* = 10^{np.log10(Ls):.4f} erg/s/Hz")
    print(f"    R² = {rc['s']['r2']:.4f}  AIC = {rc['s']['aic']:.2f}  red-χ² = {rc['s']['redchi']:.3f}")

if "Ciardullo (canonical)" in results:
    rc=results["Ciardullo (canonical)"]; pp=rc["popt"]
    Ls=L_REF*10**(-pp[1]/2.5)
    rank_c=[nm for nm,_ in ranked].index("Ciardullo (canonical)")+1
    print(f"\n  Ciardullo (canonical) — rank #{rank_c}:")
    print(f"    N(M) = {pp[0]:.2f} * exp(0.307*(M-({pp[1]:.4f}))) * (1-exp(3.0*({pp[1]:.4f}-M)))")
    print(f"    N0={pp[0]:.2f}  M*={pp[1]:.4f}  alpha=0.307 (fixed)  beta=3.0 (fixed)")
    print(f"    L* = 10^{np.log10(Ls):.4f} erg/s/Hz")
    print(f"    R² = {rc['s']['r2']:.4f}  AIC = {rc['s']['aic']:.2f}  red-χ² = {rc['s']['redchi']:.3f}")

print(f"\n  -> {OUT_VOT}")
print(f"  -> {OUT_FIG}")
