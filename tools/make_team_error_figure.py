"""Render the paper figure from saved development-analysis CSVs."""
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

REPO=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/malgun.ttf'))
args=parser.parse_args()
OUT=args.output.resolve()
assert args.font.is_file(), 'Pass --font with a Korean TrueType font path'
FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname=str(args.font))
def read(name):
    with (REPO/'reports/team_error_analysis'/name).open(encoding='utf-8') as file:return list(csv.DictReader(file))
classes=read('class_summary.csv');subjects=read('subject_summary.csv')
configs=['M01','C01','R01','L01'];models=['MLP','CNN','RNN','LSTM']
class_values=np.array([[float(next(r['f1_mean'] for r in classes if r['config_id']==c and int(r['label'])==label)) for c in configs] for label in range(6)])
ids=[1,6,14,21,23]
subject_values=np.array([[float(next(r['macro_f1_mean'] for r in subjects if r['config_id']==c and int(r['subject'])==sid)) for c in configs] for sid in ids])
supports=[int(next(r['support_per_seed'] for r in subjects if int(r['subject'])==sid)) for sid in ids]
fig,axs=plt.subplots(2,1,figsize=(80/25.4,88/25.4),layout='constrained',gridspec_kw={'height_ratios':[6,5]})
for ax,values,labels,title in zip(axs,[class_values,subject_values],[['걷기','계단↑','계단↓','앉기','서기','눕기'],[f'{sid} ({n})' for sid,n in zip(ids,supports)]],['(a) 행동별 F1','(b) 피험자별 Macro F1 (표본 수)']):
    image=ax.imshow(values,cmap='Greys',vmin=.5,vmax=1.0,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(4),models,fontsize=7.5)
    ax.set_yticks(range(len(labels)),labels,fontproperties=font,fontsize=7.5)
    ax.set_title(title,fontproperties=font,fontsize=8,pad=3)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,len(labels),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.6)
    ax.tick_params(which='minor',bottom=False,left=False)
    for (row,col),value in np.ndenumerate(values):
        ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=7.5,color='white' if value>=.82 else 'black')
    for spine in ax.spines.values():spine.set_visible(False)
axs[0].axhline(2.5,color='black',linewidth=.8,linestyle='--')
bar=fig.colorbar(image,ax=axs,shrink=.95,pad=.015,fraction=.045,ticks=[.5,.7,.9,1.0])
bar.ax.tick_params(labelsize=6.5,length=2)
fig.savefig(FIG/'class_subject_f1.png',dpi=600)
fig.savefig(FIG/'class_subject_f1.svg')
plt.close(fig)
p=FIG/'class_subject_f1.svg'
p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
(OUT/'FIGURE_DATA.json').write_text(json.dumps({'configs':configs,'models':models,'class_f1_seed_means':class_values.tolist(),'subject_ids':ids,'subject_support_per_seed':supports,'subject_macro_f1_seed_means':subject_values.tolist(),'width_mm':80,'height_mm':88,'dpi':600,'note':'Panel a: mean per-class F1 across three seeds. Panel b: mean six-class Macro F1 within each validation subject across the same seeds. Descriptive validation metrics, no confidence interval.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Generated class and subject F1 figure from audited CSVs')
