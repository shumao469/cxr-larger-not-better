from pathlib import Path
import argparse,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('design',HERE/'base_design.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def fig3():
 f=plt.figure(figsize=(7.2,8.6));v.title(f,3,'Benefits and failures differ across radiographic findings','Discrimination, calibrated probability error and local fitting resources')
 e=v.read('Head_effects_and_source_contrasts');bs=v.read('Local_head_summary');fit=v.read('Local_head_fit_attempts')
 v.label(f,.045,.902,'a','Discrimination change');v.label(f,.535,.902,'b','Calibrated probability error')
 f.legend([Line2D([0],[0],color=v.C[s],lw=1.2,marker='o',ms=3.5) for s in v.C],list(v.N.values()),loc='upper left',bbox_to_anchor=(.175,.876),ncol=2,frameon=False,fontsize=7,handlelength=1.3)
 f.legend([Line2D([0],[0],color=v.INK,ls='',marker='o',ms=4),Line2D([0],[0],color=v.INK,ls='',marker='D',mfc='white',ms=4)],['Unweighted','Class-balanced'],loc='upper left',bbox_to_anchor=(.61,.876),ncol=2,frameon=False,fontsize=6.8,handlelength=1,columnspacing=1)
 for met,x in [('AUROC',.195),('Brier',.638)]:
  ax=f.add_axes([x,.635,.315,.20]);v.clean(ax)
  for j,lab in enumerate(v.LAB):
   if j%2==0:ax.axhspan(j-.47,j+.47,color=v.PALE,zorder=0)
   for src,loss,dy,marker,filled in [('chexpert','unweighted',-.285,'o',True),('chexpert','balanced',-.095,'D',False),('nih','unweighted',.095,'o',True),('nih','balanced',.285,'D',False)]:
    r=e[e.contrast.eq(src)&e.head_loss.eq(loss)&e.finding.eq(lab)&e.metric.eq(met)].iloc[0]
    mul=1000 if met=='Brier' else 1
    v.interval(ax,mul*r.effect,mul*r.lower,mul*r.upper,j+dy,v.C[src],marker,filled)
  ax.axvline(0,c=v.MUTED,lw=.8,ls='--');ax.set_ylim(3.52,-.52)
  ax.set_yticks(range(4),['Cardiomegaly','Pleural\neffusion','Atelectasis','Consolidation'] if met=='AUROC' else v.SHORT,fontsize=6.8)
  if met=='AUROC':ax.set_xlim(-.14,.105);ax.set_xticks([-.1,-.05,0,.05,.1]);ax.set_xlabel('Δ AUROC · higher is better',fontsize=7)
  else:ax.set_xlim(-23,6);ax.set_xticks([-20,-10,0]);ax.set_xlabel('Δ Brier × 1,000 · lower is better',fontsize=7)
  ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
 f.text(.195,.581,'Local minus original head · 2,000 fitting images · 95% paired assessment intervals',fontsize=6.7,color=v.MUTED)
 v.legend_sources(f,.545,.563)
 for j,(lab,(x,y)) in enumerate(zip(v.LAB,[(.14,.386),(.63,.386),(.14,.212),(.63,.212)])):
  v.label(f,x-.095,y+.140,'cdef'[j],v.LONG[j],8.5)
  ax=f.add_axes([x,y,.32,.105]);v.clean(ax,'y')
  for src in v.C:
   z=bs[bs.source.eq(src)&bs.finding.eq(lab)&bs.head_loss.eq('unweighted')].sort_values('head_budget')
   ax.plot(z.head_budget,z.AUROC,marker='o',ms=3.2,lw=1.2,c=v.C[src])
  ax.set_xlim(150,2100);ax.set_xticks([250,500,1000,2000],['250','500','1k','2k']);ax.tick_params(labelsize=6.5)
  ax.set_ylabel('AUROC',fontsize=7);ax.set_xlabel('Local head-fitting images',fontsize=6.6,labelpad=2)
  lim,ticks=[((.92,.957),[.92,.94,.95]),((.970,.981),[.970,.975,.980]),((.80,.88),[.80,.84,.88]),((.85,.92),[.86,.89,.92])][j]
  ax.set_ylim(*lim);ax.set_yticks(ticks)
 v.label(f,.045,.145,'g','Fitting outcomes by local budget',8.5)
 ax=f.add_axes([.195,.050,.52,.072]);ax.set_xlim(0,144);ax.set_ylim(3.55,-.55)
 failcols=[v.GREEN,'#CB8772','#E3BBAC']
 for j,b in enumerate([250,500,1000,2000]):
  z=fit[fit.head_budget.eq(b)]
  nums=[int(z.status.eq('fitted').sum())]+[int((z.finding.eq(l)&z.status.ne('fitted')).sum()) for l in ['Atelectasis','Consolidation']]
  assert sum(nums)==len(z)==144
  left=0
  for n,col in zip(nums,failcols):
   if n:
    ax.barh(j,n,left=left,height=.6,color=col,edgecolor='white',lw=.7)
    ax.text(left+n/2,j,str(n),ha='center',va='center',fontsize=6.1,color='white' if col==v.GREEN else v.INK)
   left+=n
 ax.set_yticks(range(4),['250','500','1,000','2,000'],fontsize=6.5);ax.set_xticks([]);ax.tick_params(length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 f.text(.12,.066,'Fitting\nimages',ha='center',fontsize=6.5,color=v.MUTED)
 for j,(col,name) in enumerate(zip(failcols,['Successful fit','AT: class-count failure','CO: class-count failure'])):
  f.add_artist(Rectangle((.75,.109-j*.019),.012,.008,transform=f.transFigure,facecolor=col,lw=0))
  f.text(.769,.108-j*.019,name,fontsize=6.3,va='center',color=v.MUTED)
 f.text(.195,.031,'144 requests per budget; CM and PE each fitted successfully in 36/36 requests at every budget.',fontsize=6.3,color=v.MUTED)
 f.text(.045,.010,'Intervals condition on fitted models/maps. Curves show means of successful unweighted fits. AT: atelectasis; CO: consolidation.',fontsize=6.1,color=v.MUTED)
 v.save(f,3,'a,b, Paired mean changes with unadjusted 95% assessment-bootstrap intervals; source is encoded by color and head loss by marker. Brier differences are multiplied by 1,000. c–f, Mean AUROC of successful unweighted head fits. g, All 144 binary requests at each fitting budget, partitioned into successful fits and class-count failures by finding. CM and PE each fit in 36/36 requests at every budget; all failures concern AT or CO. No failed requests are excluded from these denominators.', ['Head_effects_and_source_contrasts.csv','Local_head_summary.csv','Local_head_fit_attempts.csv'])

def fig4():
 f=plt.figure(figsize=(7.2,8.6));v.title(f,4,'Training-scale gains persist across training controls','Model architecture, sample size and optimization budget examined together')
 reg=v.read('Model_registry_and_training');m=v.read('Consensus_macro_metrics').merge(reg[['run_id','training_images']],on='run_id',validate='one_to_one');g=v.read('Paired_scale_gains')
 specs=[('matched','densenet121','-','o','DenseNet-121 · 20k updates'),('convergence','densenet121','--','s','DenseNet-121 · source stopping'),('convergence','resnet34',':','^','ResNet-34 · source stopping')]
 v.label(f,.045,.902,'a','Training design across five data scales')
 ax=f.add_axes([.07,.716,.88,.149]);ax.axis('off');ax.set_xlim(0,100);ax.set_ylim(0,100)
 v.txt(ax,0,97,'MODEL AND TRAINING RULE',6.4,color=v.MUTED)
 for j,l in enumerate(['1k','5k','10k','50k','Full']):v.txt(ax,50+j*9,97,l,7,ha='center',color=v.MUTED)
 v.txt(ax,98,97,'MODELS',6.4,ha='right',color=v.MUTED)
 for i,(sch,arch,ls,mark,name) in enumerate(specs):
  y=72-i*28;v.txt(ax,0,y+3,name,8)
  ax.plot([47,89],[y-2,y-2],color=v.GRID,lw=1)
  count=0
  for j,size in enumerate(['1000','5000','10000','50000','all']):
   n=int((reg.schedule.eq(sch)&reg.architecture.eq(arch)&reg['size'].eq(size)).sum());count+=n
   ax.scatter(50+j*9,y-2,s=64 if n else 26,marker=mark if n else 'o',facecolor=v.C['chexpert'] if n else 'white',edgecolor=v.C['chexpert'] if n else v.GRID,lw=1,zorder=3)
   if n:ax.text(50+j*9,y-2,str(n),ha='center',va='center',fontsize=6,color='white')
  ax.text(97,y-2,str(count),ha='right',va='center',fontsize=11,fontweight='bold',color=v.INK)
 f.text(.07,.696,'Each included scale: 2 sources × 3 seeds. Total: 66 source models.',fontsize=7,color=v.MUTED)
 v.label(f,.045,.660,'b','CheXpert · external discrimination');v.label(f,.535,.660,'c','NIH · external discrimination')
 for src,x in [('chexpert',.135),('nih',.635)]:
  ax=f.add_axes([x,.427,.31,.191]);v.clean(ax,'y')
  for sch,arch,ls,mark,name in specs:
   z=m[m.source.eq(src)&m.schedule.eq(sch)&m.architecture.eq(arch)].groupby('size').agg(x=('training_images','mean'),y=('AUROC','mean'),sd=('AUROC','std')).sort_values('x')
   ax.fill_between(z.x,z.y-z.sd,z.y+z.sd,color=v.C[src],alpha=.07)
   ax.plot(z.x,z.y,c=v.C[src],ls=ls,marker=mark,ms=3.8,lw=1.35,alpha=1 if sch=='matched' else .68)
   ax.errorbar(z.x,z.y,yerr=z.sd,fmt='none',ecolor=v.C[src],alpha=.45,capsize=1.7,lw=.65)
  ax.set_xscale('log');ax.set_xticks([1000,10000,100000],['1k','10k','100k']);ax.set_ylim(.62,.97);ax.set_yticks([.65,.75,.85,.95]);ax.set_xlabel('Training images (log scale)',fontsize=7);ax.set_ylabel('Macro-average AUROC',fontsize=7)
 f.legend([Line2D([0],[0],color=v.INK,ls=s[2],marker=s[3],ms=3.5,lw=1.1) for s in specs],['DenseNet · 20k updates','DenseNet · source stop','ResNet · source stop'],loc='upper left',bbox_to_anchor=(.13,.382),ncol=3,frameon=False,fontsize=6.4,columnspacing=2.4,handlelength=2)
 v.label(f,.045,.323,'d','Observed update budgets');v.label(f,.535,.323,'e','Smallest-to-full AUROC gains')
 ax=f.add_axes([.20,.100,.265,.174]);v.clean(ax)
 rng=np.random.default_rng(984)
 for i,(sch,arch,ls,mark,name) in enumerate(specs):
  for src,dy in [('chexpert',-.18),('nih',.18)]:
   z=reg[reg.source.eq(src)&reg.schedule.eq(sch)&reg.architecture.eq(arch)]
   vals=z.updates.to_numpy()/1000;yy=np.full(len(vals),i+dy)+rng.uniform(-.065,.065,len(vals))
   ax.scatter(vals,yy,c=v.C[src],s=11,alpha=.55,marker=mark,zorder=3)
   ax.plot([vals.min(),vals.max()],[i+dy,i+dy],c=v.C[src],alpha=.3,lw=.9)
 ax.set_yticks(range(3),['DenseNet\n20k updates','DenseNet\nsource stop','ResNet\nsource stop'],fontsize=6.5);ax.set_ylim(2.5,-.5);ax.set_xlim(0,25);ax.set_xticks([0,5,10,15,20,25]);ax.set_xlabel('Successful updates (×1,000)',fontsize=6.8)
 ax=f.add_axes([.715,.100,.235,.174]);v.clean(ax)
 for i,(sch,arch,ls,mark,name) in enumerate(specs):
  for src,dy in [('chexpert',-.15),('nih',.15)]:
   z=g[g.source.eq(src)&g.schedule.eq(sch)&g.architecture.eq(arch)].iloc[0]
   v.interval(ax,z.gain,z.bootstrap_97_5_lower,z.bootstrap_97_5_upper,i+dy,v.C[src],mark)
 ax.set_yticks(range(3),['DenseNet\n20k updates','DenseNet\nsource stop','ResNet\nsource stop'],fontsize=6.5);ax.set_ylim(2.5,-.5);ax.set_xlim(0,.25);ax.set_xticks([0,.1,.2]);ax.set_xlabel('Δ AUROC · 97.5% intervals',fontsize=6.8)
 v.legend_sources(f,.35,.063)
 v.footer(f,'b–c: mean ± sample SD across three seeds. d: each dot is one source model, across all scales; jitter separates coincident dots.\ne: paired assessment-image bootstrap intervals, conditional on fitted models. Equal updates retain differences in exposure per image.')
 v.save(f,4,'a, Design and counts for 66 source models; each included scale contains two sources and three seeds. b,c, Macro-average AUROC means and sample SDs. d, All 66 observed successful update budgets, with fixed visual jitter and source color; line spans the observed minimum and maximum within source and condition. e, Paired smallest-to-full gains with 97.5% assessment-bootstrap intervals, conditional on fitted models. Matching updates does not match composition or nominal presentations per image.', ['Model_registry_and_training.csv','Consensus_macro_metrics.csv','Paired_scale_gains.csv'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--assets',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    v.D=args.source_data;v.ROOT=args.output;v.F=args.output/'Figures';v.F.mkdir(exist_ok=True)
    if args.assets:v.A=args.assets;v.figure1()
    v.figure2();fig3();fig4()
    with PdfPages(args.output/'Reproduced_figures.pdf') as pdf:
        for fig in v.figures:pdf.savefig(fig)
