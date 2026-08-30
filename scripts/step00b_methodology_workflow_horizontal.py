"""Single-row version of the pipeline schematic used in the paper.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
plt.rcParams.update({'font.family':'DejaVu Sans'})

NAVY='#12305a'; BLUE='#2f6fb3'; TEAL='#1b8a8f'; AMBER='#d1802a'; RED='#b4402f'
GREY='#5b6470'; LIGHT='#eef2f7'; EDGE='#c3cdd9'

W,H=17.8,5.2
fig,ax=plt.subplots(figsize=(W,H)); ax.set_xlim(0,100); ax.set_ylim(0,29); ax.axis('off')

def stage(x, w, title, colour, items, y0=3.6, h=19.4):
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle="round,pad=0.25,rounding_size=0.8",
                 fc='white',ec=EDGE,lw=1.1,zorder=2))
    ax.add_patch(FancyBboxPatch((x,y0+h-3.0),w,3.0,boxstyle="round,pad=0.25,rounding_size=0.8",
                 fc=colour,ec=colour,lw=1.1,zorder=3))
    ax.text(x+w/2, y0+h-1.5, title, ha='center', va='center', color='white',
            fontsize=10.2, fontweight='bold', zorder=4)
    yy=y0+h-4.6
    for code,txt,note in items:
        ax.text(x+0.7, yy, code, ha='left', va='top', fontsize=7.4, color=colour, fontweight='bold', zorder=4)
        ax.text(x+3.6, yy, txt, ha='left', va='top', fontsize=8.2, color='#1a1f26', zorder=4)
        if note:
            ax.text(x+3.6, yy-1.45, note, ha='left', va='top', fontsize=7.1, color=GREY, style='italic', zorder=4)
            yy-=3.55
        else:
            yy-=2.35
    return x+w

def arrow(x1,x2,y=13.3):
    ax.add_patch(FancyArrowPatch((x1+0.4,y),(x2-0.4,y),arrowstyle='-|>',mutation_scale=15,
                 lw=1.5,color=GREY,zorder=5))

gap=1.9
x=0.6
w1,w2,w3,w4,w5,w6=14.2,16.4,15.2,14.8,16.4,15.6

e=stage(x,w1,'1.  INPUT DATA',NAVY,[
  ('01a','Unified optical catalogue','Reid & Parker + R14 photometry  →  679 PNe'),
  ('','MeerKAT 1.295 GHz mosaic','3σ and 5σ Aegean catalogues'),
  ('','ASKAP–EMU 888 MHz','54,612 sources'),
  ('','Spitzer SAGE 8 μm','605 PNe with photometry'),
])
arrow(e,e+gap); x=e+gap

e=stage(x,w2,'2.  CROSS-MATCH',BLUE,[
  ('02a','Reid × MeerKAT','4.5″ nearest neighbour  →  188'),
  ('02b','Reid × ASKAP','4.5″ accept; wider pairs held for review  →  61'),
  ('','Chance-coincidence estimate','~10 spurious expected in 679'),
])
arrow(e,e+gap); x=e+gap

e=stage(x,w3,'3.  RECOVERY & QA',TEAL,[
  ('03a','Split unmatched sources','23 outside MeerKAT footprint'),
  ('03b','Visual inspection of MeerKAT','peak + local RMS  →  32 recovered'),
  ('03c','ASKAP four-panel QA','ASKAP / Hα / [O III] / 8 μm'),
])
arrow(e,e+gap); x=e+gap

e=stage(x,w4,'4.  MERGE & STATUS',AMBER,[
  ('04a','MeerKAT catalogue + sky maps','164 MeerKAT-only'),
  ('04b','Multi-survey union','56 both  +  5 ASKAP-only  →  225'),
  ('04c','HASH V/163 status check','208 listed  (200 T, 8 P)'),
])
arrow(e,e+gap); x=e+gap

e=stage(x,w5,'5.  DIAGNOSTICS',RED,[
  ('05a','Combined spectral index','MeerKAT sub-bands + ASKAP → 112 reliable'),
  ('06a','MIR/radio ratio','measured at 888, 997 and 1295 MHz'),
  ('06b','Evidence-based grading','28 H  /  90 P  /  97 p  /  10 Q'),
])
arrow(e,e+gap); x=e+gap

e=stage(x,w6,'6.  RADIO PNLF',NAVY,[
  ('07a','Luminosities and binning','220 with 1.295 GHz flux'),
  ('07b','Ten-model fits, ranked by AIC','Ciardullo + empirical forms'),
  ('07c','Sample scenarios S1–S5','M* = −4.16 mag (clean sample)'),
])

ax.text(0.6,27.6,'Radio planetary-nebula pipeline for the Large Magellanic Cloud',
        fontsize=13.4,fontweight='bold',color=NAVY,ha='left')
ax.text(0.6,25.4,'Each block corresponds to one numbered script in the public repository; '
        'counts are the values obtained in this work.',
        fontsize=8.6,color=GREY,ha='left')
ax.plot([0.6,99.4],[24.2,24.2],lw=0.9,color=EDGE)
# ax.text(99.4,1.1,'github.com/<user>/lmc-radio-pne',fontsize=7.6,color=GREY,ha='right',style='italic')
# changed 2026-08-30: placeholder <user> replaced by the real repository owner
ax.text(99.4,1.1,'github.com/okhattab/lmc-radio-pne',fontsize=7.6,color=GREY,ha='right',style='italic')

fig.tight_layout(pad=0.4)
out=os.environ.get("PNBASE",".")+"/04_Figures/step00b_methodology_workflow_horizontal.pdf"
fig.savefig(out); print("wrote",out)
