"""Draw a baseline architecture plate and a full-width diagnostic figure."""
from pathlib import Path
import argparse,csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--paper-layout',choices=['full','compact'],default='full',help='full: three wide panels; compact: two single-column diagnostic panels')
parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/malgun.ttf'))
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
OUT=args.output.resolve();FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname=str(args.font))
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.linewidth':.6,'svg.fonttype':'path','savefig.facecolor':'white'})
def read(path):
    with (repo/path).open(encoding='utf-8-sig') as file:return list(csv.DictReader(file))
baseline=read(Path('reports/team_comparison/baseline_summary.csv'))
subjects=read(Path('reports/team_error_analysis/subject_summary.csv'))
paired=read(Path('reports/subject_composition/paired_seed_gaps.csv'))
configs=['M01','C01','R01','L01'];models=['MLP','CNN','RNN','LSTM']
assert [r['config_id'] for r in baseline]==configs
assert all(r['evaluation_split']=='development validation' and r['test_evaluated']=='False' for r in baseline)
ids=[1,6,14,21,23]
values=np.array([[float(next(r['macro_f1_mean'] for r in subjects if r['config_id']==c and int(r['subject'])==sid)) for c in configs] for sid in ids])
counts=[int(next(r['support_per_seed'] for r in subjects if int(r['subject'])==sid)) for sid in ids]
order=['none','1','6','14','21','23']
gaps=[float(next(r['mean_macro_f1_difference'] for r in paired if r['left_config']=='C01' and r['right_config']=='L01' and r['removed_subject']==s)) for s in order]
data={'scope':'fixed-configuration development validation; no refitting or official test','new_training':False,'test_evaluated':False,'figures':[]}
def save(fig,stem,width,height,meaning,content):
    fig.savefig(FIG/(stem+'.png'),dpi=600)
    fig.savefig(FIG/(stem+'.svg'))
    plt.close(fig)
    svg=FIG/(stem+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    data['figures'].append({'file':stem+'.png','width_mm':width,'height_mm':height,'dpi':600,'meaning':meaning,'values':content,'sha256':hashlib.sha256((FIG/(stem+'.png')).read_bytes()).hexdigest()})

fig,ax=plt.subplots(figsize=(80/25.4,58/25.4))
fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
blue='#275975';red='#A54448';grey='#66717C'
ax.add_patch(Rectangle((.15,.897),.82,.079,facecolor='#EEF3F6',edgecolor=blue,linewidth=.65))
ax.text(.56,.937,'공통 입력 128시점 × 9채널',ha='center',va='center',fontsize=7.4,color='#26343E')
lanes=[('MLP / M01',.735,['Flatten\n1,152','Dense\n128','Dense\n64','6 logits']),
       ('CNN / C01',.54,['Conv1D\n64 / k5','Conv1D\n128 / k3','Conv1D\n128 / k3','GAP +\nDense64','6 logits']),
       ('RNN / R01',.345,['SimpleRNN 32\n마지막 상태','6 logits']),
       ('LSTM / L01',.15,['LSTM 64\n마지막 상태','Dense\n32','6 logits'])]
for name,y,blocks in lanes:
    accent=red if name.startswith('LSTM') else blue
    ax.text(.014,y+.033,name.split(' / ')[0],fontsize=8.2,weight='bold',color=accent,ha='left')
    ax.text(.014,y-.032,name.split(' / ')[1],fontsize=6.4,color=grey,ha='left')
    n=len(blocks);start=.155;end=.972;gap=.021;width=(end-start-gap*(n-1))/n;height=.124
    for index,label in enumerate(blocks):
        x=start+index*(width+gap)
        face='#F8ECEC' if name.startswith('LSTM') else '#F3F6F8'
        ax.add_patch(Rectangle((x,y-height/2),width,height,facecolor=face,edgecolor=accent,linewidth=.6))
        ax.text(x+width/2,y,label,ha='center',va='center',fontsize=6.4 if n==5 else 7.0,linespacing=1.16)
        if index<n-1:
            ax.add_patch(FancyArrowPatch((x+width,y),(x+width+gap,y),arrowstyle='-|>',mutation_scale=5,linewidth=.6,color=accent))
    if name.startswith('CNN'):
        ax.text(.555,y-.103,'첫 두 Conv 뒤 Pool 2 / Dense 뒤 Dropout 0.3',ha='center',fontsize=5.8,color=grey)
ax.text(.56,.835,'같은 사람 분할 · 학습 통계 표준화 · 3 seeds',ha='center',fontsize=6.9,color=grey)
ax.text(.51,.018,'RNN·LSTM: 구간마다 상태 0 초기화 / 모든 출력은 logits',ha='center',fontsize=6.25,color=grey)
save(fig,'baseline_architectures',80,58,'Architecture of the actual M01/C01/R01/L01 baselines; layer labels and states are schematic, not measured activations.',{'input_shape':[128,9],'configs':configs,'fit_subjects':16,'val_subjects':5,'fit_windows':5577,'val_windows':1775,'seeds':[2026,2027,2028],'hidden_dense_activation':'ReLU','cnn_pool':'size 2 after first two conv layers','cnn_dense_dropout':.3,'other_baseline_dropout':0,'recurrent_initial_state':'zero independently for each window','output':'6 logits'})

if args.paper_layout=='compact':
    fig,axs=plt.subplots(2,1,figsize=(80/25.4,76/25.4),layout='constrained',gridspec_kw={'height_ratios':[1.1,1]})
    ax=axs[0]
    ax.imshow(values,cmap='Blues',vmin=.5,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(4),models,fontsize=7.2)
    ax.set_yticks(range(5),[f'{sid} ({n})' for sid,n in zip(ids,counts)],fontsize=7.0)
    ax.set_title('(a) 피험자별 Macro F1',fontsize=8.0,weight='bold',loc='left',pad=5)
    ax.set_ylabel('피험자 ID (n)',fontsize=7.2,labelpad=2)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',length=0)
    for (row,col),value in np.ndenumerate(values):
        ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=7.2,color='white' if value>=.82 else '#253541')
    ax.add_patch(Rectangle((-.5,1.5),4,1,facecolor='none',edgecolor=red,linewidth=1.0))
    for spine in ax.spines.values():spine.set_visible(False)
    ax=axs[1]
    ax.barh(range(6),gaps,color=[red if value<0 else blue for value in gaps],height=.55)
    ax.axvline(0,color='#34434C',linewidth=.7)
    ax.set_xlim(-.012,.029);ax.set_ylim(5.65,-.65)
    ax.set_yticks(range(6),['전체 5명','1 제외','6 제외','14 제외','21 제외','23 제외'],fontsize=7.0)
    ax.set_xticks([-.01,0,.01,.02],['-.01','0','.01','.02'])
    for index,value in enumerate(gaps):
        ax.text(value+(.0006 if value>=0 else -.0006),index,f'{value:+.3f}',va='center',ha='left' if value>=0 else 'right',fontsize=6.8,color=red if value<0 else '#253541')
    ax.set_xlabel('평균 F1 차이: CNN - LSTM',fontsize=7.2,labelpad=2)
    ax.set_title('(b) 검증 구성 민감도',fontsize=8.0,weight='bold',loc='left',pad=5)
    ax.tick_params(axis='x',labelsize=7.0,length=2);ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',alpha=.17,linewidth=.6)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    save(fig,'subject_diagnostics_compact',80,76,'Single-column two-panel development diagnostic: within-subject six-class F1 seed means and all five omission CNN-LSTM gaps; no retraining. Baseline metrics retained as provenance for the accompanying table, not plotted here.',{'baseline':[{k:r[k] for k in ['model','config_id','parameters','macro_f1_mean','macro_f1_sample_sd']} for r in baseline],'configs':configs,'subject_ids':ids,'support_per_seed':counts,'subject_macro_f1_seed_means':values.tolist(),'omission_order':order,'cnn_minus_lstm_mean':gaps,'no_refitting':True,'headline_subjects':ids})
    (OUT/'VISUAL_FIGURE_DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Created a baseline architecture and a single-column two-panel diagnostic figure.')
    raise SystemExit(0)

fig=plt.figure(figsize=(170/25.4,54/25.4),layout='constrained')
grid=fig.add_gridspec(1,3,width_ratios=[1.03,1.10,.96],wspace=.11)
axs=[fig.add_subplot(grid[0,i]) for i in range(3)]
colors=['#76818A',blue,'#B39A73',red]
ax=axs[0]
for row,color in zip(baseline,colors):
    x=int(row['parameters']);y=float(row['macro_f1_mean']);sd=float(row['macro_f1_sample_sd'])
    ax.errorbar(x,y,yerr=sd,fmt='o',markersize=4,color=color,elinewidth=.9,capsize=2)
    off={'M01':(-2,-13),'C01':(-2,13),'R01':(4,-10),'L01':(-2,13)}[row['config_id']]
    ax.annotate(row['model'],(x,y),xytext=off,textcoords='offset points',ha='right' if off[0]<0 else 'left',fontsize=7.8)
ax.set_xscale('log');ax.set_xlim(1000,245000);ax.set_ylim(.60,1.0)
ax.set_xticks([1000,10000,100000],['1k','10k','100k']);ax.set_yticks([.6,.7,.8,.9,1.0])
ax.set_xlabel('파라미터 수 (로그 축)',fontsize=8,labelpad=4)
ax.set_ylabel('Macro F1',fontsize=8,labelpad=2)
ax.set_title('(a) 성능·모델 규모',fontsize=9,weight='bold',loc='left',pad=9)
ax.text(.02,.97,'3 seeds: 평균 ± SD',transform=ax.transAxes,fontsize=7,va='top',color=grey)
ax.tick_params(labelsize=7.4,length=2);ax.grid(axis='y',alpha=.17,linewidth=.6)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)

ax=axs[1]
im=ax.imshow(values,cmap='Blues',vmin=.5,vmax=1,aspect='auto',interpolation='nearest')
ax.set_xticks(range(4),models,fontsize=7.3)
ax.set_yticks(range(5),[f'{sid}\n(n={n})' for sid,n in zip(ids,counts)],fontsize=7.2)
ax.set_title('(b) 피험자별 편차',fontsize=9,weight='bold',loc='left',pad=9)
ax.set_ylabel('피험자 ID',fontsize=8,labelpad=2)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',length=0)
for (row,col),value in np.ndenumerate(values):
    ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=7.5,color='white' if value>=.82 else '#253541')
ax.add_patch(Rectangle((-.5,1.5),4,1,facecolor='none',edgecolor=red,linewidth=1.1))
for spine in ax.spines.values():spine.set_visible(False)
ax.set_xlabel('각 셀: 피험자 내부 Macro F1',fontsize=7.4,labelpad=5)

ax=axs[2]
ax.barh(range(6),gaps,color=[red if value<0 else blue for value in gaps],height=.56)
ax.axvline(0,color='#34434C',linewidth=.7)
ax.set_xlim(-.010,.028);ax.set_ylim(5.7,-.7)
ax.set_yticks(range(6),['전체 5명','1 제외','6 제외','14 제외','21 제외','23 제외'],fontsize=7.5)
ax.set_xticks([-.01,0,.01,.02],['-.01','0','.01','.02'])
for index,value in enumerate(gaps):
    if value<0:
        ax.text(value/2,index+.39,f'{value:+.3f}',va='center',ha='center',fontsize=7.0,color=red)
    else:
        ax.text(value+.0006,index,f'{value:+.3f}',va='center',ha='left',fontsize=7.2)
ax.set_xlabel('평균 F1 차이: CNN - LSTM',fontsize=7.9,labelpad=4)
ax.set_title('(c) 검증 구성 민감도',fontsize=9,weight='bold',loc='left',pad=9)
ax.tick_params(axis='x',labelsize=7.3,length=2);ax.tick_params(axis='y',length=0)
ax.grid(axis='x',alpha=.17,linewidth=.6)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.spines['left'].set_visible(False)
save(fig,'research_summary',170,54,'Full-width three-panel development diagnostic. (a) parameters vs Macro F1 with three-seed sample SD; (b) within-subject six-class F1 seed means, subject 14 outlined; (c) mean CNN-LSTM gaps in all five omission subsets, without retraining. All five people retained in headline results.',{'baseline':[{k:r[k] for k in ['model','config_id','parameters','macro_f1_mean','macro_f1_sample_sd']} for r in baseline],'configs':configs,'subject_ids':ids,'support_per_seed':counts,'subject_macro_f1_seed_means':values.tolist(),'omission_order':order,'cnn_minus_lstm_mean':gaps,'no_refitting':True,'headline_subjects':ids})
(OUT/'VISUAL_FIGURE_DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Created an architecture plate and a 170 mm three-panel research figure.')
