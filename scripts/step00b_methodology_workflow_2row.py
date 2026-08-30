"""Two-row version of the pipeline schematic used in the paper.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
plt.rcParams.update({'font.family':'DejaVu Sans'})

NAVY='#12305a'; BLUE='#2f6fb3'; TEAL='#1b8a8f'; AMBER='#cf7f28'; RED='#b4402f'
GREY='#59626e'; EDGE='#c3cdd9'

fig,ax=plt.subplots(figsize=(13.2,8.0))
# ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
# changed 2026-08-30: the row-1 to row-2 return connector is drawn at x=100.8,
# which the old xlim clipped at the right edge
ax.set_xlim(-1.5,103); ax.set_ylim(0,100); ax.axis('off')

BW=29.2      # box width
GAPX=3.6     # horizontal gap
BH=36.0      # box height
def stage(x,y,title,colour,items):
    ax.add_patch(FancyBboxPatch((x,y),BW,BH,boxstyle="round,pad=0.4,rounding_size=1.1",
                 fc='white',ec=EDGE,lw=1.2,zorder=2))
    ax.add_patch(FancyBboxPatch((x,y+BH-5.4),BW,5.4,boxstyle="round,pad=0.4,rounding_size=1.1",
                 fc=colour,ec=colour,lw=1.2,zorder=3))
    ax.text(x+BW/2,y+BH-2.7,title,ha='center',va='center',color='white',
            fontsize=13.5,fontweight='bold',zorder=4)
    yy=y+BH-8.4
    for code,txt,note in items:
        ax.text(x+1.4,yy,code,ha='left',va='top',fontsize=9.4,color=colour,fontweight='bold',zorder=4)
        ax.text(x+6.4,yy,txt,ha='left',va='top',fontsize=10.6,color='#161b22',zorder=4)
        ax.text(x+6.4,yy-2.9,note,ha='left',va='top',fontsize=9.2,color=GREY,style='italic',zorder=4)
        yy-=7.4

def harrow(x1,x2,y):
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle='-|>',mutation_scale=19,lw=2.0,color=GREY,zorder=5))

ROW1=48.0; ROW2=7.0
X=[3.0, 3.0+BW+GAPX, 3.0+2*(BW+GAPX)]

stage(X[0],ROW1,'1.  INPUT DATA',NAVY,[
 ('01a','Unified optical catalogue','Reid & Parker + R14  →  679 PNe'),
 ('','MeerKAT 1.295 GHz mosaic','3σ and 5σ Aegean catalogues'),
 ('','ASKAP–EMU 888 MHz','54,612 sources'),
 ('','Spitzer SAGE 8 μm','605 PNe with photometry')])
harrow(X[0]+BW+0.6, X[1]-0.6, ROW1+BH/2)

stage(X[1],ROW1,'2.  CROSS-MATCH',BLUE,[
 ('02a','Reid × MeerKAT','4.5″ nearest neighbour  →  188'),
 ('02b','Reid × ASKAP','4.5″ accept, wider pairs reviewed  →  61'),
 ('','Independent matching','each survey matched to the parent'),
 ('','Chance coincidences','~10 expected across 679')])
harrow(X[1]+BW+0.6, X[2]-0.6, ROW1+BH/2)

stage(X[2],ROW1,'3.  RECOVERY & QA',TEAL,[
 ('03a','Split unmatched sources','23 outside MeerKAT footprint'),
 ('03b','Visual inspection','peak + local RMS  →  32 recovered'),
 ('03c','ASKAP four-panel QA','ASKAP / Hα / [O III] / 8 μm'),
 ('','Recoveries flagged','never define the luminosity function')])

# wrap-around connector from end of row 1 to start of row 2
yA=ROW1+BH/2; yB=ROW2+BH/2
ax.add_patch(FancyArrowPatch((X[2]+BW+0.6, yA), (X[2]+BW+3.0, yA),
             arrowstyle='-',lw=2.0,color=GREY,zorder=5))
ax.add_patch(FancyArrowPatch((X[2]+BW+3.0, yA), (X[2]+BW+3.0, (yA+yB)/2),
             arrowstyle='-',lw=2.0,color=GREY,zorder=5))
ax.add_patch(FancyArrowPatch((X[2]+BW+3.0,(yA+yB)/2), (1.0,(yA+yB)/2),
             arrowstyle='-',lw=2.0,color=GREY,zorder=5))
ax.add_patch(FancyArrowPatch((1.0,(yA+yB)/2), (1.0, yB),
             arrowstyle='-',lw=2.0,color=GREY,zorder=5))
ax.add_patch(FancyArrowPatch((1.0,yB),(X[0]-0.6,yB),arrowstyle='-|>',mutation_scale=19,lw=2.0,color=GREY,zorder=5))

stage(X[0],ROW2,'4.  MERGE & STATUS',AMBER,[
 ('04a','MeerKAT catalogue + sky maps','164 MeerKAT-only'),
 ('04b','Multi-survey union','56 both + 5 ASKAP-only  →  225'),
 ('04c','HASH V/163 status check','208 listed (200 T, 8 P)'),
 ('','Status never scored','origin traces to the same optical survey')])
harrow(X[0]+BW+0.6, X[1]-0.6, ROW2+BH/2)

stage(X[1],ROW2,'5.  DIAGNOSTICS',RED,[
 ('05a','Combined spectral index','MeerKAT sub-bands + ASKAP  →  112'),
 ('06a','MIR/radio ratio','888, 997 and 1295 MHz, never rescaled'),
 ('06b','Evidence-based grading','28 H / 90 P / 97 p / 10 Q'),
 ('','Diagnostics kept separate','combined only at the grading step')])
harrow(X[1]+BW+0.6, X[2]-0.6, ROW2+BH/2)

stage(X[2],ROW2,'6.  RADIO PNLF',NAVY,[
 ('07a','Luminosities and binning','220 with a 1.295 GHz flux'),
 ('07b','Ten models ranked by AIC','Ciardullo + empirical forms'),
 ('07c','Sample scenarios S1–S5','M* = −4.16 mag (clean sample)'),
 ('','No luminosity masking','only a sensitivity-based faint cut')])

ax.text(3.0,97.0,'Radio planetary-nebula pipeline for the Large Magellanic Cloud',
        fontsize=17.5,fontweight='bold',color=NAVY,ha='left')
ax.text(3.0,92.6,'Each block is one numbered script in the public repository; counts are the values obtained in this work.',
        fontsize=11.2,color=GREY,ha='left')
ax.plot([3.0,97.0],[89.6,89.6],lw=1.0,color=EDGE)
ax.text(97.0,1.5,'github.com/okhattab/lmc-radio-pne',fontsize=9.6,color=GREY,ha='right',style='italic')

fig.tight_layout(pad=0.3)
out=os.environ.get("PNBASE",".")+"/04_Figures/step00b_methodology_workflow_2row.pdf"
fig.savefig(out); print("wrote",out)
