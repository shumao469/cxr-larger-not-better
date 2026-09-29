"""Four figure-only revisions from a frozen aggregate snapshot. No model fitting."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'Source_Data'; A=ROOT/'Assets'; F=ROOT/'Figures'
for name in ['arial.ttf','arialbd.ttf']:
    for root in [Path('C:/Windows/Fonts'),Path('/mnt/c/Windows/Fonts')]:
        if (root/name).exists():font_manager.fontManager.addfont(str(root/name))
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,
 'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'axes.edgecolor':'#8C9AA3','xtick.color':'#465560','ytick.color':'#465560',
 'text.color':'#20333F','axes.labelcolor':'#20333F','pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#20333F'; MUTED='#6E818D'; GRID='#E4EBEF'; PALE='#F3F7F9'
C={'chexpert':'#087EA4','nih':'#D47738'}; N={'chexpert':'CheXpert','nih':'NIH'}
GREEN='#228D79'; PURPLE='#8570AC'; RED='#BE665B'
LAB=['Cardiomegaly','Pleural_Effusion','Atelectasis','Consolidation']
LONG=['Cardiomegaly','Pleural effusion','Atelectasis','Consolidation']
SHORT=['CM','PE','AT','CO']; figures=[]; descriptions={}; source_map={}

def read(n):return pd.read_csv(D/(n+'.csv'),dtype={'size':str})
def title(fig,n,text,sub):
    fig.text(.045,.973,f'{n:02d}',fontsize=17,fontweight='bold',color=C['chexpert'],va='top')
    fig.text(.115,.971,text,fontsize=12,fontweight='bold',va='top')
    fig.text(.115,.941,sub,fontsize=7.4,color=MUTED,va='top')
    fig.add_artist(Line2D([.045,.965],[.920,.920],color=GRID,lw=.9,transform=fig.transFigure))
def label(fig,x,y,letter,text,fs=9):
    fig.text(x,y,letter,fontsize=12,fontweight='bold',va='top')
    fig.text(x+.034,y-.002,text,fontsize=fs,fontweight='bold',va='top')
def clean(ax,axis='x'):
    ax.grid(axis=axis,color=GRID,lw=.65);ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.65)
def legend_sources(fig,x,y):
    fig.legend([Line2D([0],[0],color=C[s],marker='o',lw=1.2,ms=4) for s in C],
      list(N.values()),bbox_to_anchor=(x,y),loc='upper left',ncol=2,frameon=False,
      borderaxespad=0,handlelength=1.5,columnspacing=1.5,fontsize=7.3)
def interval(ax,x,lo,hi,y,col,marker='o',filled=True):
    ax.plot([lo,hi],[y,y],color=col,lw=1.2,zorder=3)
    ax.scatter(x,y,marker=marker,s=24,facecolor=col if filled else 'white',edgecolor=col,lw=1.1,zorder=4)
def footer(fig,text):fig.text(.045,.022,text,fontsize=6.6,color=MUTED,va='bottom',linespacing=1.4)
def save(fig,n,description,inputs):
    name=f'Figure_{n}';figures.append(fig);descriptions[name]=description;source_map[name]=inputs
    for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=360)
    fig.savefig(F/f'{name}.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    print(name,flush=True)
def arrow(ax,x1,y1,x2,y2,col=MUTED,lw=1,style='-|>'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=8,color=col,lw=lw))
def txt(ax,x,y,s,size=8,**kw):ax.text(x,y,s,fontsize=size,va='top',**kw)
def box(ax,x,y,w,h,col=GRID,fill=PALE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.015',
      edgecolor=col,facecolor=fill,lw=.8))

def figure1():
    fig=plt.figure(figsize=(7.2,8.6))
    title(fig,1,'Where local data can change chest radiograph AI',
          'Image representation, probability estimation and clinical operating thresholds')
    label(fig,.045,.902,'a','Three image resources with distinct analytical roles')
    cohorts=[('chexpert','CheXpert',C['chexpert'],'87,939 training images','12,668 validation images'),
             ('nih','NIH ChestX-ray14',C['nih'],'79,481 training images','11,342 validation images'),
             ('vindr','VinDr-CXR',GREEN,'15,000 development images','3,000 consensus images')]
    for i,(asset,n,col,l1,l2) in enumerate(cohorts):
        x=.05+i*.312
        im=Image.open(A/(asset+'_example.png'))
        ax=fig.add_axes([x,.706,.135,.160]);ax.imshow(im);ax.axis('off')
        fig.text(x+.15,.858,n,fontsize=9,fontweight='bold',color=col)
        fig.text(x+.15,.824,l1.replace(' images','\nimages'),fontsize=7.6,linespacing=1.25)
        fig.text(x+.15,.768,l2.replace(' images','\nimages'),fontsize=7.6,linespacing=1.25)
        fig.add_artist(Line2D([x,x+.282],[.698,.698],color=col,lw=1.8,transform=fig.transFigure))
    fig.text(.05,.683,'Illustrative radiographs from the source datasets; displayed without diagnostic annotation.',fontsize=6.7,color=MUTED)
    label(fig,.045,.651,'b','Separate interventions on the model and its outputs')
    ax=fig.add_axes([.05,.328,.91,.294]);ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    # DenseNet schematic: layer counts correspond to DenseNet-121, not fitted feature maps.
    txt(ax,5,55,'224 × 224\nRGB input',6.7,color=MUTED,ha='center')
    ia=fig.add_axes([.052,.484,.09,.103]);ia.imshow(Image.open(A/'chexpert_example.png'));ia.axis('off')
    txt(ax,17,99,'DenseNet-121 encoder',9,fontweight='bold')
    for j,(x,w,h,layers) in enumerate([(17,5,27,6),(28,5,24,12),(39,5,21,24),(50,5,18,16)]):
        for k in reversed(range(4)):
            ax.add_patch(Rectangle((x+k*.8,60+k*.9),w,h,facecolor='#EDF5F7',edgecolor='#66A7B7',lw=.75))
        txt(ax,x+3,55,str(layers),6.7,ha='center',color=MUTED)
        if j<3:arrow(ax,x+8.5,72,x+10.5,72)
    arrow(ax,10,72,16,72)
    txt(ax,36,46,'Dense-block layers (schematic)',6.6,ha='center',color=MUTED)
    arrow(ax,59,72,64,72)
    box(ax,65,61,8,22,PURPLE,'#F4F1F8');txt(ax,69,77,'1,024',7,ha='center');txt(ax,69,68,'features',5.9,ha='center')
    txt(ax,69,55,'Pooling',6.6,ha='center',color=MUTED)
    arrow(ax,74,72,79,72)
    # Head may be the original multi-output head or locally fitted per-finding logistic heads.
    box(ax,80,58,19,28,GREEN,'#EFF7F3');txt(ax,89.5,82,'Classifier head',8,ha='center',fontweight='bold')
    txt(ax,89.5,72,r'$p=\sigma(w^{T}z+b)$',9,ha='center')
    txt(ax,89.5,61,'Four findings',6.5,ha='center',color=GREEN)
    ax.plot([16,74],[40,40],color=C['chexpert'],lw=1.2)
    txt(ax,16,35,'Source training: update encoder + head',7.5,color=C['chexpert'])
    txt(ax,16,25,'Local head fitting: freeze encoder; refit linear heads',7.5,color=GREEN)
    txt(ax,16,14,r'$z=f_{\theta}(x)$',10,color=INK)
    # Probability and operating-threshold analysis both start from the head score.
    fig.add_artist(Line2D([.878,.878,.260],[.494,.337,.337],color='#A6B6C0',lw=.9,transform=fig.transFigure))
    for xx in [.260,.790]:
        fig.add_artist(FancyArrowPatch((xx,.337),(xx,.320),arrowstyle='-|>',mutation_scale=8,color=MUTED,lw=.9,transform=fig.transFigure))
    # Parallel output branches are drawn in a dedicated strip.
    ax2=fig.add_axes([.05,.255,.91,.064]);ax2.axis('off');ax2.set_xlim(0,100);ax2.set_ylim(0,1)
    for x,w,col in [(0,46,PURPLE),(53,47,RED)]:box(ax2,x,.07,w,.86,col,'white')
    txt(ax2,2,.82,'Probability calibration',8,fontweight='bold',color=PURPLE)
    txt(ax2,2,.40,r'$q=\sigma(a\,\mathrm{logit}(p)+c)$',10)
    txt(ax2,55,.82,'Threshold selection',8,fontweight='bold',color=RED)
    txt(ax2,55,.40,r'$\hat{y}=1\{p\geq t\}$',10)
    label(fig,.045,.230,'c','Separate fitting, calibration and assessment images')
    labels=[('LOCAL HEAD FITTING','3,000-image subset','250–2,000 images used','Majority-reader labels',GREEN),
            ('LOCAL CALIBRATION','1,000 consensus images','42 effusion positives','Probability maps + thresholds',PURPLE),
            ('FIXED ASSESSMENT','2,000 consensus images','69 effusion positives','1,931 effusion negatives',RED)]
    for i,(k,main,sub,detail,col) in enumerate(labels):
        x=.05+i*.314
        fig.add_artist(Rectangle((x,.088),.287,.114,transform=fig.transFigure,facecolor=colors.to_rgba(col,.055),edgecolor='none'))
        fig.add_artist(Rectangle((x,.088),.004,.114,transform=fig.transFigure,facecolor=col,edgecolor='none'))
        fig.text(x+.012,.186,k,fontsize=7.1,fontweight='bold',color=col)
        fig.text(x+.012,.160,main,fontsize=8,fontweight='bold')
        fig.text(x+.012,.137,sub,fontsize=7.2)
        fig.text(x+.012,.114,detail,fontsize=6.9,color=MUTED)
    footer(fig,'CM: cardiomegaly · PE: pleural effusion · AT: atelectasis · CO: consolidation\nRetrospective public-data analysis; image separation is established, while patient independence remains unverified.')
    save(fig,1,'Real dataset radiographs are illustrative only. DenseNet-121 block schematic is conceptual; no activation or saliency map is simulated. Probability calibration and threshold estimation are parallel uses of original or refitted scores. Local fitting uses a fixed subset of the 15,000-image development resource; calibration and assessment use disjoint image partitions of the consensus release. These are established model components, not a new architecture.', ['Model_registry_and_training.csv','Assets/provenance.json'])

def figure2():
    fig=plt.figure(figsize=(7.2,8.6));title(fig,2,'Local adaptation changes missed findings and false flags','Pleural effusion · identical assessment images across strategies')
    ops=read('Clinical_operating_points');q=read('Joint_uncertainty_summary');p=read('Joint_paired_strategy_differences')
    h=read('Local_head_operating_intervals');cloud=read('Local_head_operating_bootstrap')
    labels=['Original · source','Original · local empirical','Original · order statistic','Local head · empirical','Local head · order statistic']
    label(fig,.045,.902,'a','Missed positive labels');label(fig,.535,.902,'b','False-positive flags')
    legend_sources(fig,.55,.873)
    for metric,pos,xlim in [('fn',[.275,.664,.205,.180],(0,16)),('fp',[.665,.664,.280,.180],(0,840))]:
        ax=fig.add_axes(pos);clean(ax)
        for j in range(5):
            if j%2==0:ax.axhspan(j-.47,j+.47,color=PALE,zorder=0)
        for src,dy in [('chexpert',-.13),('nih',.13)]:
            a=ops[ops.source.eq(src)].reset_index(drop=True)
            assert len(a)==5
            for j,row in a.iterrows():
                ax.plot([0,row[metric]],[j+dy,j+dy],color=C[src],alpha=.24,lw=2)
                ax.scatter(row[metric],j+dy,s=19,c=C[src],zorder=3)
                ax.annotate(f'{row[metric]:.1f}',(row[metric],j+dy),xytext=(4,0),textcoords='offset points',va='center',fontsize=6.2,color=C[src])
        ax.set_ylim(4.55,-.55);ax.set_xlim(*xlim)
        ax.set_yticks(range(5),labels if metric=='fn' else ['']*5,fontsize=6.6)
        ax.set_xlabel('Missed / 69 positives' if metric=='fn' else 'False flags / 1,931 negatives',fontsize=7)
        ax.set_xticks([0,5,10,15] if metric=='fn' else [0,200,400,600,800]);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.text(.275,.618,'Counts are averages over fitted models; local heads use 2,000 development images.',fontsize=6.7,color=MUTED)
    label(fig,.045,.583,'c','Probability error');label(fig,.535,.583,'d','Threshold tradeoff')
    ax=fig.add_axes([.145,.402,.330,.142]);clean(ax)
    for i,src in enumerate(C):
        for method,dy,filled in [('raw',-.15,False),('logistic',.15,True)]:
            r=q[q.source.eq(src)&q.method.eq(method)&q.metric.eq('Brier')].iloc[0]
            interval(ax,r.estimate,r.lower,r.upper,i+dy,C[src],filled=filled)
    ax.set_yticks([0,1],list(N.values()),fontsize=7);ax.set_ylim(1.6,-.6);ax.set_xlim(0,.11);ax.set_xticks([0,.025,.05,.075,.1],['0','.025','.050','.075','.100']);ax.set_xlabel('Brier score · lower is better',fontsize=7)
    ax.legend([Line2D([0],[0],color=INK,marker='o',mfc='white',ls=''),Line2D([0],[0],color=INK,marker='o',ls='')],['Raw','Local calibration'],frameon=False,fontsize=6.4,ncol=2,loc='lower right',bbox_to_anchor=(1,1.0),handletextpad=.4,columnspacing=.7)
    ax=fig.add_axes([.65,.402,.295,.142]);clean(ax)
    for i,src in enumerate(C):
        for met,dy,mark in [('sensitivity',-.16,'o'),('specificity',.16,'s')]:
            r=p[p.source.eq(src)&p.left.eq('empirical')&p.right.eq('guarded')&p.metric.eq(met)].iloc[0]
            interval(ax,100*r.difference,100*r.lower,100*r.upper,i+dy,C[src],mark)
    ax.axvline(0,color=MUTED,ls='--',lw=.8);ax.set_xlim(-38,21);ax.set_ylim(1.6,-.6);ax.set_yticks([0,1],list(N.values()),fontsize=7)
    ax.set_xlabel('Order statistic − empirical (pp)',fontsize=7)
    ax.legend([Line2D([0],[0],color=INK,marker=m,ls='') for m in ['o','s']],['Sensitivity','Specificity'],frameon=False,fontsize=6.4,ncol=2,loc='lower right',bbox_to_anchor=(1,1.0),handletextpad=.4,columnspacing=.7)
    label(fig,.045,.331,'e','CheXpert · head-refitting tradeoff');label(fig,.535,.331,'f','NIH · head-refitting tradeoff')
    for src,x in [('chexpert',.135),('nih',.635)]:
        ax=fig.add_axes([x,.097,.31,.188]);clean(ax,'both')
        c=cloud[cloud.source.eq(src)&cloud.head_loss.eq('unweighted')&cloud.method.eq('guarded')&cloud.replicate.gt(0)].dropna(subset=['sensitivity','specificity'])
        ax.axhspan(-25,0,color='#FBF0ED',zorder=0)
        ax.scatter(100*c.specificity,100*c.sensitivity,s=6,c=C[src],alpha=.17,edgecolors='none',rasterized=True)
        rs=h[h.source.eq(src)&h.head_loss.eq('unweighted')&h.method.eq('guarded')]
        sx=rs[rs.metric.eq('specificity')].iloc[0];sy=rs[rs.metric.eq('sensitivity')].iloc[0]
        xx=100*sx.difference_head_minus_original;yy=100*sy.difference_head_minus_original
        ax.plot([100*sx.lower,100*sx.upper],[yy,yy],color=C[src],lw=1.5)
        ax.plot([xx,xx],[100*sy.lower,100*sy.upper],color=C[src],lw=1.5)
        ax.scatter(xx,yy,marker='D',s=37,c=C[src],edgecolors='white',lw=.8,zorder=5)
        ax.axhline(0,color=MUTED,ls='--',lw=.8);ax.axvline(0,color=MUTED,ls='--',lw=.8)
        ax.set_xlim(-5,40);ax.set_ylim(-20,10);ax.set_xticks([0,10,20,30,40]);ax.set_yticks([-20,-10,0,10])
        ax.set_xlabel('Change in specificity (pp)',fontsize=7)
        ax.set_ylabel('Change in sensitivity (pp)',fontsize=7)
        ax.text(.96,.96,'Both improve',transform=ax.transAxes,fontsize=6.1,color=GREEN,va='top',ha='right')
        ax.text(.97,.06,'Sensitivity decreases',transform=ax.transAxes,fontsize=6.1,color=RED,ha='right')
    footer(fig,'e–f: local minus original head under the same order-statistic rule. Dots: joint bootstrap draws; cross: marginal 95% intervals.\n42 calibration positives · 69 assessment positives · fixed fitted heads · label-conditioned counts, rather than measured clinical workload')
    save(fig,2,'a,b, Mean missed positive labels and false-positive flags across five strategies. Original-head values average three source seeds; local-head values average nine fits. c,d, Joint calibration/assessment 95% bootstrap intervals. e,f, Joint bootstrap changes after unweighted local-head fitting under the order-statistic rule; diamond is the observed paired mean, with marginal 95% interval cross, not a joint confidence region. Feasible resamples are retained, including tails. The bound is feasible in 494/500 original-strategy resamples and 496/500 head-effect resamples. All counts refer to the same fixed image-label assessment set.', ['Clinical_operating_points.csv','Joint_uncertainty_summary.csv','Joint_paired_strategy_differences.csv','Local_head_operating_intervals.csv','Local_head_operating_bootstrap.csv'])

def figure3():
    fig=plt.figure(figsize=(7.2,8.6));title(fig,3,'Benefits and failures differ across radiographic findings','Joint assessment of discrimination, probability error and local fitting resources')
    e=read('Head_effects_and_source_contrasts');bs=read('Local_head_summary');fit=read('Local_head_fit_attempts')
    cols=[('chexpert','unweighted'),('chexpert','balanced'),('nih','unweighted'),('nih','balanced')]
    cmap=colors.LinearSegmentedColormap.from_list('effects',[RED,'#FAFBFC',GREEN])
    label(fig,.045,.902,'a','Discrimination change');label(fig,.535,.902,'b','Calibrated probability error')
    for met,x,lim in [('AUROC',.17,.08),('Brier',.65,.020)]:
        ax=fig.add_axes([x,.656,.295,.197]);values=np.zeros((4,4))
        for i,l in enumerate(LAB):
            for j,(src,loss) in enumerate(cols):values[i,j]=e[e.contrast.eq(src)&e.head_loss.eq(loss)&e.finding.eq(l)&e.metric.eq(met)].iloc[0].effect
        norm=colors.TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
        # Green denotes improvement for each metric; the printed values retain their original signs.
        ax.imshow(values if met=='AUROC' else -values,cmap=cmap,norm=norm,aspect='auto')
        for i,l in enumerate(LAB):
            for j,(src,loss) in enumerate(cols):
                row=e[e.contrast.eq(src)&e.head_loss.eq(loss)&e.finding.eq(l)&e.metric.eq(met)].iloc[0]
                def f(v):
                    return (f'{1000*v:+.1f}' if met=='Brier' else f'{v:+.3f}'.replace('0.','.')).replace('-','−')
                ax.text(j,i-.11,f(row.effect),ha='center',va='center',fontsize=7.5,fontweight='bold',color=INK)
                ax.text(j,i+.23,f'{f(row.lower)}, {f(row.upper)}',ha='center',va='center',fontsize=5.45,color=INK)
        ax.set_xticks(range(4),['Unw.','Bal.','Unw.','Bal.'],fontsize=6.7)
        ax.set_yticks(range(4),SHORT,fontsize=7.3);ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
        for sp in ax.spines.values():sp.set_visible(False)
        ax.text(.25,1.065,'CheXpert',ha='center',transform=ax.transAxes,fontsize=7.3,color=C['chexpert'],fontweight='bold')
        ax.text(.75,1.065,'NIH',ha='center',transform=ax.transAxes,fontsize=7.3,color=C['nih'],fontweight='bold')
        ax.set_xlabel('Δ AUROC · higher is better' if met=='AUROC' else 'Δ Brier × 1,000 · lower is better',fontsize=7)
    fig.text(.17,.603,'Cell: mean change and 95% interval. Green: improvement; coral: deterioration.',fontsize=6.4,color=MUTED)
    fig.text(.17,.587,'Color intensity is scaled separately for each metric. Unw.: unweighted; Bal.: class-balanced.',fontsize=6.4,color=MUTED)
    # Four compact learning curves use common x coordinates; axes remain finding-specific and explicit.
    coords=[(.14,.386),(.63,.386),(.14,.190),(.63,.190)]
    for j,(l,(x,y)) in enumerate(zip(LAB,coords)):
        label(fig,x-.095,y+.157,'cdef'[j],LONG[j],8.5)
        ax=fig.add_axes([x,y,.32,.12]);clean(ax,'y')
        for src in C:
            d=bs[bs.source.eq(src)&bs.finding.eq(l)&bs.head_loss.eq('unweighted')].sort_values('head_budget')
            ax.plot(d.head_budget,d.AUROC,marker='o',ms=3.2,lw=1.2,c=C[src])
        ax.set_xlim(150,2100);ax.set_xticks([250,500,1000,2000],['250','500','1k','2k']);ax.tick_params(axis='both',labelsize=6.5)
        ax.set_ylabel('AUROC',fontsize=7);ax.set_xlabel('Local head-fitting images',fontsize=6.7,labelpad=2)
        if l=='Cardiomegaly':ax.set_ylim(.92,.957);ax.set_yticks([.92,.94,.95])
        if l=='Pleural_Effusion':ax.set_ylim(.970,.981);ax.set_yticks([.970,.975,.980])
        if l=='Atelectasis':ax.set_ylim(.80,.88);ax.set_yticks([.80,.84,.88])
        if l=='Consolidation':ax.set_ylim(.85,.92);ax.set_yticks([.86,.89,.92])
    legend_sources(fig,.52,.575)
    label(fig,.045,.125,'g','Fitting availability',8)
    ax=fig.add_axes([.32,.067,.62,.061]);ax.axis('off');ax.set_xlim(0,4);ax.set_ylim(-.2,4.2)
    for i,l in enumerate(LAB):
        ax.text(-.12,3.5-i,SHORT[i],ha='right',va='center',fontsize=6.4)
        for j,b in enumerate([250,500,1000,2000]):
            d=fit[fit.finding.eq(l)&fit.head_budget.eq(b)];n=int(d.status.eq('fitted').sum());den=len(d);frac=n/den
            ax.add_patch(Rectangle((j+.025,3.04-i),.95,.91,facecolor=colors.to_rgba(GREEN,.10+.72*frac),edgecolor='white',lw=.8))
            ax.text(j+.5,3.5-i,f'{n}/{den}',ha='center',va='center',fontsize=6,color='white' if frac>.65 else INK)
    for j,b in enumerate(['250','500','1,000','2,000']):ax.text(j+.5,4.24,b,ha='center',va='bottom',fontsize=6.2)
    fig.text(.08,.088,'Successful / requested\nbinary fits',fontsize=6.9,color=MUTED)
    footer(fig,'a–b: 2,000 fitting images; intervals condition on fitted models and maps. c–f: unweighted heads; means of successful fits.\nCM: cardiomegaly · PE: pleural effusion · AT: atelectasis · CO: consolidation. Failed fits remain in the denominators in g.')
    save(fig,3,'a,b, Effects of local head fitting at 2,000 development images: mean and unadjusted 95% paired assessment-bootstrap interval. Brier differences and interval endpoints are multiplied by 1,000 for legibility. Green denotes improvement according to each metric, so AUROC and Brier use opposite color directions with separate intensity scales. Both heads receive separate local logistic calibration before Brier comparison. c–f, Mean discrimination across available unweighted local heads and fitting budgets; missing combinations indicate failed fits. Vertical scales are finding-specific and labelled. g, Successful over requested binary fits across both source backbones and both local losses. Fitting availability is a resource property, not clinical model accuracy.', ['Head_effects_and_source_contrasts.csv','Local_head_summary.csv','Local_head_fit_attempts.csv'])

def figure4():
    fig=plt.figure(figsize=(7.2,8.6));title(fig,4,'Training-scale gains persist across training controls','Matched update counts and an alternative architecture contextualize local adaptation')
    reg=read('Model_registry_and_training');m=read('Consensus_macro_metrics');g=read('Paired_scale_gains')
    m=m.merge(reg[['run_id','training_images']],on='run_id',validate='one_to_one')
    specs=[('matched','densenet121','-','o','DenseNet · 20k updates'),('convergence','densenet121','--','s','DenseNet · source stop'),('convergence','resnet34',':','^','ResNet · source stop')]
    label(fig,.045,.902,'a','Controlled training design');label(fig,.535,.902,'b','Smallest-to-full AUROC gains')
    legend_sources(fig,.68,.878)
    ax=fig.add_axes([.05,.702,.41,.152]);ax.axis('off');ax.set_xlim(0,100);ax.set_ylim(0,100)
    txt(ax,0,98,'MODEL / SELECTION',6,color=MUTED)
    for j,v in enumerate(['1k','5k','10k','50k','Full']):txt(ax,49+j*10,98,v,6.6,ha='center',color=MUTED)
    for i,(sch,arch,ls,mark,lab) in enumerate(specs):
        y=72-i*28
        txt(ax,0,y+4,lab.replace(' · ','\n'),6.8)
        for j,size in enumerate(['1000','5000','10000','50000','all']):
            n=int((reg.schedule.eq(sch)&reg.architecture.eq(arch)&reg['size'].eq(size)).sum())
            ax.scatter(49+j*10,y-3,s=46,facecolor=GREEN if n else 'white',edgecolor=GREEN if n else GRID,lw=.8)
            if n:ax.text(49+j*10,y-3,str(n),ha='center',va='center',color='white',fontsize=6)
    fig.text(.05,.675,'66 source models · 2 sources × 3 seeds at each included scale',fontsize=6.7,color=MUTED)
    ax=fig.add_axes([.715,.702,.230,.151]);clean(ax)
    for j,(sch,arch,ls,mark,lab) in enumerate(specs):
        for src,dy in [('chexpert',-.13),('nih',.13)]:
            r=g[g.source.eq(src)&g.schedule.eq(sch)&g.architecture.eq(arch)].iloc[0]
            interval(ax,r.gain,r.bootstrap_97_5_lower,r.bootstrap_97_5_upper,j+dy,C[src],mark)
    ax.set_yticks([0,1,2],['DenseNet\n20k updates','DenseNet\nsource stop','ResNet\nsource stop'],fontsize=6.4)
    ax.set_ylim(2.5,-.5);ax.set_xlim(0,.25);ax.set_xticks([0,.1,.2]);ax.set_xlabel('Δ AUROC · 97.5% intervals',fontsize=6.7)
    label(fig,.045,.625,'c','CheXpert · external discrimination');label(fig,.535,.625,'d','NIH · external discrimination')
    for src,x in [('chexpert',.14),('nih',.63)]:
        ax=fig.add_axes([x,.410,.32,.174]);clean(ax,'y')
        for sch,arch,ls,mark,lab in specs:
            a=m[m.source.eq(src)&m.schedule.eq(sch)&m.architecture.eq(arch)].groupby('size').agg(x=('training_images','mean'),y=('AUROC','mean'),sd=('AUROC','std')).sort_values('x')
            alpha=1 if sch=='matched' else .65
            ax.fill_between(a.x,a.y-a.sd,a.y+a.sd,color=C[src],alpha=.075)
            ax.plot(a.x,a.y,ls=ls,marker=mark,ms=3.5,lw=1.3,c=C[src],alpha=alpha)
            ax.errorbar(a.x,a.y,yerr=a.sd,fmt='none',ecolor=C[src],alpha=.45,lw=.6,capsize=1.6)
        ax.set_xscale('log');ax.set_xticks([1000,10000,100000],['1k','10k','100k']);ax.set_ylim(.62,.97);ax.set_yticks([.65,.75,.85,.95]);ax.set_ylabel('Macro-average AUROC',fontsize=7);ax.set_xlabel('Training images (log scale)',fontsize=7)
    fig.legend([Line2D([0],[0],color=INK,ls=s[2],marker=s[3],ms=3.4,lw=1.1) for s in specs],[s[4] for s in specs],loc='upper left',bbox_to_anchor=(.13,.367),ncol=3,fontsize=6.3,frameon=False,columnspacing=1.1,handlelength=2.4)
    label(fig,.045,.302,'e','CheXpert · observed update budget');label(fig,.535,.302,'f','NIH · observed update budget')
    for src,x in [('chexpert',.14),('nih',.63)]:
        ax=fig.add_axes([x,.091,.32,.171]);clean(ax,'y')
        for sch,arch,ls,mark,lab in specs:
            a=reg[reg.source.eq(src)&reg.schedule.eq(sch)&reg.architecture.eq(arch)].groupby('size').agg(x=('training_images','mean'),y=('updates','mean'),lo=('updates','min'),hi=('updates','max')).sort_values('x')
            ax.plot(a.x,a.y/1000,ls=ls,marker=mark,ms=3.5,lw=1.2,c=C[src],alpha=1 if sch=='matched' else .65)
            ax.errorbar(a.x,a.y/1000,yerr=np.array([a.y-a.lo,a.hi-a.y])/1000,fmt='none',ecolor=C[src],alpha=.5,lw=.7,capsize=2)
        ax.set_xscale('log');ax.set_xticks([1000,10000,100000],['1k','10k','100k']);ax.set_ylim(0,30);ax.set_yticks([0,10,20,30]);ax.set_ylabel('Successful updates (×1,000)',fontsize=7);ax.set_xlabel('Training images (log scale)',fontsize=7)
    footer(fig,'b: paired assessment-image bootstrap intervals, conditional on fitted models. c–d: mean ± sample SD across three seeds.\ne–f: mean and full seed range. Source stopping is a selection rule; equal update counts retain differences in exposure per image.')
    save(fig,4,'a, Number of source models at each included scale and control condition; six means two sources times three seeds. b, Paired smallest-to-full discrimination gains with 97.5% assessment-bootstrap intervals, conditional on fitted models. The two matched-update DenseNet contrasts were the earlier primary comparisons. c,d, Learning curves on the same 2,000 assessment images; means, sample-SD bars and bands across three source seeds. e,f, Observed successful optimizer updates, with mean and minimum-to-maximum seed range. Source stopping uses best-checkpoint evaluation; matched-update models use the final checkpoint. Matching update counts leaves differences in nominal exposure, composition and augmentation.', ['Model_registry_and_training.csv','Consensus_macro_metrics.csv','Paired_scale_gains.csv'])

def main():
    F.mkdir(exist_ok=True)
    for func in [figure1,figure2,figure3,figure4]:func()
    with PdfPages(ROOT/'Four_Main_Figures.pdf') as pdf:
        for fig in figures:pdf.savefig(fig)
    (ROOT/'Panel_descriptions.md').write_text('# Figure-only design notes\n\nThe manuscript and its existing legends are unchanged. These notes document the expanded panels for the next text revision.\n\n'+'\n\n'.join(f'## {k}\n\n{v}' for k,v in descriptions.items()),encoding='utf-8')
    (ROOT/'Audit/Figure_source_map.json').write_text(json.dumps(source_map,indent=2),encoding='utf-8')
    (ROOT/'Audit/Figure_manifest.json').write_text(json.dumps([{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(F.glob('*'))],indent=2),encoding='utf-8')
    for f in figures:plt.close(f)
if __name__=='__main__':main()
