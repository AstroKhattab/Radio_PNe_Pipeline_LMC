"""step00b_methodology_workflow_horizontal.py — two-row pipeline figure."""
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

stage(X[0],ROW1,'1.  PARENT CATALOGUE',NAVY,[
 ('1','HASH V/163, LMC domain','545 true + 162 probable  →  707 PNe'),
 ('','Reid & Parker cross-match','RP number and optical class  →  605'),
 ('','Spitzer SAGE 8 μm','265 PNe with photometry'),
 ('','138 LMC entries set aside','clusters, SNRs, H II regions')])
harrow(X[0]+BW+0.6, X[1]-0.6, ROW1+BH/2)

stage(X[1],ROW1,'2.  CROSS-MATCH',BLUE,[
 ('2a','Parent × MeerKAT 1.295 GHz','4.5″ nearest neighbour  →  225'),
 ('2b','Parent × ASKAP 888 MHz','4.5″ accept, wider pairs flagged  →  82'),
 ('','One parent, both surveys','the bookkeeping closes on 707'),
 ('','Median offset 0.76″','1.6 chance matches expected')])
harrow(X[1]+BW+0.6, X[2]-0.6, ROW1+BH/2)

stage(X[2],ROW1,'3.  INSPECT & QA',TEAL,[
 ('3a','Split the unmatched','460 inspected, 22 outside the mosaic'),
 ('3b','Inspect and extract','peak + local RMS  →  24 recovered'),
 ('3b2','Aperture correction','enclosed-flux factor applied'),
 ('3e','ASKAP four-panel QA','ASKAP / Hα / [O III] / 8 μm')])

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

stage(X[0],ROW2,'4.  THREE CRITERIA',AMBER,[
 ('4a','Spectral index','128 reliable  →  45 thermal, 88 steep'),
 ('4b','MIR / radio ratio','888, 997 and 1295 MHz, never rescaled'),
 ('4c','Radio flux ceiling','2.2 mJy at 49.59 kpc  →  7 above'),
 ('','No optical input','the criteria stay independent')])
harrow(X[0]+BW+0.6, X[1]-0.6, ROW2+BH/2)

stage(X[1],ROW2,'5.  CLASSIFICATION',RED,[
 ('5e','Scored out of what is measurable','72 High / 104 Possible / 70 Weak'),
 ('5e','Hard rejections','above the ceiling, or offset > 4.5″  →  8'),
 ('5e','Tested against HASH and Reid','34% vs 6% reach the highest class'),
 ('','Optical class never used','which is what makes the test valid')])
harrow(X[1]+BW+0.6, X[2]-0.6, ROW2+BH/2)

stage(X[2],ROW2,'6.  RADIO PNLF',NAVY,[
 ('6a','Luminosities and binning','242 enter the fit, 13 bins'),
 ('6a','Ten models ranked by AIC','Ciardullo + empirical forms'),
 ('6b','Canonical Ciardullo','M* = −4.40 mag, red-χ² = 1.53'),
 ('','Two masks only','the 5σ limit and the flux ceiling')])

ax.text(3.0,97.0,'Radio planetary-nebula pipeline for the Large Magellanic Cloud',
        fontsize=17.5,fontweight='bold',color=NAVY,ha='left')
ax.text(3.0,92.6,'Each block is one numbered script in the public repository; counts are the values obtained in this work.',
        fontsize=11.2,color=GREY,ha='left')
ax.plot([3.0,97.0],[89.6,89.6],lw=1.0,color=EDGE)
ax.text(97.0,1.5,'github.com/okhattab/lmc-radio-pne',fontsize=9.6,color=GREY,ha='right',style='italic')

fig.tight_layout(pad=0.3)
out=os.environ.get("PNBASE",".")+"/05_Figures/step00b_methodology_workflow_2row.pdf"
fig.savefig(out); print("wrote",out)
