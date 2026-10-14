"""Render descriptive validation composition plots from audited CSVs."""
from pathlib import Path
import argparse,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/malgun.ttf'))
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
def read(name):
    with (repo/'reports/subject_composition'/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
summary=read('summary.csv');gaps=read('paired_seed_gaps.csv')
order=['none','1','6','14','21','23']
configs=['M01','C01','R01','L01'];names=['MLP','CNN','RNN','LSTM']
values={c:[float(next(r['macro_f1_mean'] for r in summary if r['config_id']==c and r['removed_subject']==s)) for s in order] for c in configs}
differences=[float(next(r['mean_macro_f1_difference'] for r in gaps if r['left_config']=='C01' and r['right_config']=='L01' and r['removed_subject']==s)) for s in order]
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname=str(args.font))
fig,axes=plt.subplots(2,1,figsize=(160/25.4,110/25.4),layout='constrained',gridspec_kw={'height_ratios':[1.45,1]})
colors=['#5A6472','#245F9B','#8E7652','#A63643']
for c,name,color in zip(configs,names,colors):
    axes[0].plot(range(6),values[c],marker='o',markersize=4,linewidth=1.1,label=name,color=color)
axes[0].set_ylim(.65,.97)
axes[0].set_ylabel('Macro F1',fontsize=8)
axes[0].set_title('(a) 같은 저장 예측의 검증 집단별 평균',fontproperties=font,fontsize=9)
axes[0].legend(ncol=4,fontsize=7,loc='lower right')
axes[0].grid(axis='y',alpha=.2)
axes[0].set_xticks(range(6),['전체 5명','1 제외','6 제외','14 제외','21 제외','23 제외'],fontproperties=font,fontsize=8)
axes[1].bar(range(6),differences,color=['#245F9B' if v>=0 else '#A63643' for v in differences],width=.55)
axes[1].axhline(0,color='black',linewidth=.7)
for index,value in enumerate(differences):
    axes[1].text(index,value+(.001 if value>=0 else -.001),f'{value:+.4f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=7.5)
axes[1].set_ylim(-.010,.028)
axes[1].set_ylabel('CNN - LSTM',fontsize=8)
axes[1].set_title('(b) 세 seed의 평균 Macro F1 차이',fontproperties=font,fontsize=9)
axes[1].set_xticks(range(6),['전체 5명','1 제외','6 제외','14 제외','21 제외','23 제외'],fontproperties=font,fontsize=8)
axes[1].grid(axis='y',alpha=.2)
for ax in axes:
    ax.tick_params(axis='y',labelsize=7)
    ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
fig.savefig(OUT/'subject_composition.png',dpi=400)
fig.savefig(OUT/'subject_composition.svg')
plt.close(fig)
svg=OUT/'subject_composition.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
(OUT/'SUBJECT_COMPOSITION_FIGURE_DATA.json').write_text(json.dumps({'configs':configs,'removed_subject_order':order,'macro_f1_mean':values,'cnn_minus_lstm_mean':differences,'headline_subjects':[1,6,14,21,23],'headline_count':1775,'seeds':[2026,2027,2028],'no_refitting':True,'post_hoc':True,'width_mm':160,'height_mm':110,'dpi':400,'note':'Each excluded-person subset contains the remaining four validation subjects. Sample count and class composition change. Not LOSO cross-validation, not independent tests, and no improvement claimed.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Generated audited composition diagnostic figure and exact data.')
