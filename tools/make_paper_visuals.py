"""Create the three main-paper figures from audited development results."""
from pathlib import Path
import argparse,csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/malgun.ttf'))
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
OUT=args.output.resolve();FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname=str(args.font))
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
def read(path):
    with (repo/path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
baseline=read(Path('reports/team_comparison/baseline_summary.csv'))
subjects=read(Path('reports/team_error_analysis/subject_summary.csv'))
composition=read(Path('reports/subject_composition/paired_seed_gaps.csv'))
configs=['M01','C01','R01','L01'];models=['MLP','CNN','RNN','LSTM']
assert [r['config_id'] for r in baseline]==configs
assert all(r['evaluation_split']=='development validation' and r['test_evaluated']=='False' for r in baseline)
data={'scope':'development validation only; existing fixed configurations and saved predictions','new_training':False,'test_evaluated':False,'figures':[]}
def save(fig,stem,width,height,meaning,values):
    fig.savefig(FIG/(stem+'.png'),dpi=600)
    fig.savefig(FIG/(stem+'.svg'))
    plt.close(fig)
    svg=FIG/(stem+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    data['figures'].append({'file':stem+'.png','width_mm':width,'height_mm':height,'dpi':600,'meaning':meaning,'values':values,'sha256':hashlib.sha256((FIG/(stem+'.png')).read_bytes()).hexdigest()})

fig,ax=plt.subplots(figsize=(80/25.4,25/25.4))
fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
boxes=[(.01,.56,.27,.38,'제공 시계열\n128×9 / 6행동'),(.365,.56,.27,.38,'사람 단위 분할\n학습 16 / 검증 5'),(.72,.56,.27,.38,'학습 통계 표준화\n평균·SD 1회 적용'),(.72,.06,.27,.38,'고정 네 구성\n3seed씩 12회'),(.365,.06,.27,.38,'검증 손실로\nepoch 선택'),(.01,.06,.27,.38,'성능·규모 분석\n피험자 진단')]
for x,y,w,h,label in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006,rounding_size=.018',edgecolor='#566675',facecolor='#F2F5F7',linewidth=.6))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontproperties=font,fontsize=6.6,linespacing=1.16)
for start,end in [((.28,.75),(.36,.75)),((.635,.75),(.715,.75)),((.855,.55),(.855,.45)),((.715,.25),(.64,.25)),((.36,.25),(.285,.25))]:
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,linewidth=.6,color='#566675'))
save(fig,'experiment_flow',80,25,'Verified development workflow, with validation also used for epoch selection. Arrows indicate procedure only; no independent test evaluation.',{'input_shape':[128,9],'classes':6,'fit_subjects':16,'val_subjects':5,'fit_windows':5577,'val_windows':1775,'seeds':[2026,2027,2028],'formal_baseline_runs':12,'normalization':'saved fit-only mean and standard deviation, applied once','checkpoint_selection':'minimum validation loss'})

fig,ax=plt.subplots(figsize=(80/25.4,47/25.4),layout='constrained')
colors=['#646B74','#246498','#9B8260','#A73747']
offsets={'M01':(-4,-14),'C01':(-5,15),'R01':(7,-10),'L01':(-5,15)}
for row,color in zip(baseline,colors):
    x=int(row['parameters']);y=float(row['macro_f1_mean']);sd=float(row['macro_f1_sample_sd'])
    ax.errorbar(x,y,yerr=sd,fmt='o',markersize=3.5,color=color,elinewidth=.9,capsize=2)
    dx,dy=offsets[row['config_id']]
    ax.annotate(row['model'],(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=7)
ax.set_xscale('log');ax.set_xlim(1000,230000);ax.set_ylim(.60,1.005)
ax.set_xticks([1000,10000,100000],['1k','10k','100k'])
ax.set_yticks([.6,.7,.8,.9,1.0])
ax.set_xlabel('파라미터 수 (로그 축)',fontproperties=font,fontsize=7.5,labelpad=2)
ax.set_ylabel('Macro F1',fontsize=7.5,labelpad=2)
ax.set_title('세 seed 평균 ± 표본 SD',fontproperties=font,fontsize=7.5,pad=3)
ax.tick_params(labelsize=7,length=2);ax.grid(axis='y',alpha=.2,linewidth=.5)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
save(fig,'performance_size',80,47,'Mean six-class development Macro F1 versus trainable parameter count; x-axis logarithmic, bars show three-seed sample SD, not confidence intervals or device timing.',[{k:r[k] for k in ['model','config_id','parameters','macro_f1_mean','macro_f1_sample_sd']} for r in baseline])

ids=[1,6,14,21,23]
values=np.array([[float(next(r['macro_f1_mean'] for r in subjects if r['config_id']==c and int(r['subject'])==sid)) for c in configs] for sid in ids])
counts=[int(next(r['support_per_seed'] for r in subjects if int(r['subject'])==sid)) for sid in ids]
order=['none','1','6','14','21','23']
gaps=[float(next(r['mean_macro_f1_difference'] for r in composition if r['left_config']=='C01' and r['right_config']=='L01' and r['removed_subject']==s)) for s in order]
fig,axes=plt.subplots(2,1,figsize=(80/25.4,76/25.4),layout='constrained',gridspec_kw={'height_ratios':[1.55,1]})
ax=axes[0]
im=ax.imshow(values,cmap='Greys',vmin=.5,vmax=1,interpolation='nearest',aspect='auto')
ax.set_xticks(range(4),models,fontsize=7)
ax.set_yticks(range(5),[f'{s} ({n})' for s,n in zip(ids,counts)],fontsize=7)
ax.set_title('(a) 피험자별 Macro F1 (구간 수)',fontproperties=font,fontsize=7.7,pad=3)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',length=0)
for (row,col),value in np.ndenumerate(values):
    ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=7,color='white' if value>=.82 else 'black')
for spine in ax.spines.values():spine.set_visible(False)
bar=fig.colorbar(im,ax=ax,fraction=.04,pad=.02,ticks=[.5,1]);bar.ax.tick_params(labelsize=6,length=2)
ax=axes[1]
ax.bar(range(6),gaps,color=['#246498' if v>=0 else '#A73747' for v in gaps],width=.58)
ax.axhline(0,color='black',linewidth=.6)
ax.set_ylim(-.012,.032);ax.set_yticks([-.01,0,.01,.02,.03]);ax.tick_params(labelsize=6.5,length=2)
for index,value in enumerate(gaps):
    ax.text(index,value+(.001 if value>=0 else -.001),f'{value:+.3f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=6.1)
ax.set_xticks(range(6),['전체','1','6','14','21','23'],fontsize=7)
ax.set_ylabel('CNN - LSTM',fontsize=7,labelpad=1)
ax.set_xlabel('제외 피험자 (전체는 제외 없음)',fontproperties=font,fontsize=6.8,labelpad=2)
ax.set_title('(b) 검증 구성별 평균 F1 차이',fontproperties=font,fontsize=7.7,pad=4)
ax.grid(axis='y',alpha=.2,linewidth=.5)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
save(fig,'subject_diagnostics',80,76,'Panel a: six-class Macro F1 inside each subject, then mean over three seeds. Panel b: CNN minus LSTM mean differences on all five subjects or each of five omission subsets, without retraining; sample and class composition also change. Headline score retains all five.',{'configs':configs,'subject_ids':ids,'support_per_seed':counts,'subject_macro_f1_seed_means':values.tolist(),'omission_order':order,'cnn_minus_lstm_mean':gaps,'no_refitting':True,'headline_subjects':ids})
(OUT/'VISUAL_FIGURE_DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Generated workflow, performance-size chart, and subject diagnostic panels from audited results.')
