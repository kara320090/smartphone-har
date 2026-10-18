"""Render source-derived figures for the v9 paper (no synthetic result data)."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':7,'axes.unicode_minus':False,'svg.fonttype':'none'})
source=ROOT/'reports/paper_controls_20261019/all_mapping_summary.csv'
with source.open(encoding='utf-8') as f:data=list(csv.DictReader(f))
with (ROOT/'reports/paper_controls_20261019/mapping_rank_summary.csv').open(encoding='utf-8') as f:ranks=list(csv.DictReader(f))
def save(fig,name):
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=400,facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(3.15,1.92));fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,t,fc='#eef3f8',size=7):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=.014',lw=.65,edgecolor='#3d5265',facecolor=fc))
    ax.text(x,y,t,ha='center',va='center',fontsize=size,linespacing=1.4)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':.8,'color':'#34495e','shrinkA':1,'shrinkB':1})
box(.5,.90,.77,.13,'학습 통계로 표준화한 센서 구간 128×9')
box(.25,.65,.44,.18,'시계열 입력\nMLP · CNN · RNN · LSTM')
box(.75,.65,.43,.18,'채널 평균·표준편차 18개\n학습 자료로 특징 표준화')
arrow(.35,.827,.25,.75);arrow(.65,.827,.75,.75)
box(.25,.38,.30,.14,'6종 예측');box(.61,.38,.25,.14,'LR6: 6종');box(.88,.38,.21,.14,'LR3: 3상태')
arrow(.25,.55,.25,.46);arrow(.69,.55,.61,.46);arrow(.81,.55,.88,.46)
box(.5,.115,.94,.14,'3상태: 이동 / 앉거나 서 있음 / 누워 있음',fc='#e5f2ee',size=7)
arrow(.25,.30,.25,.19);arrow(.61,.30,.61,.19);arrow(.88,.30,.88,.19)
ax.text(.41,.245,'6종→3상태 매핑',ha='center',va='center',fontsize=6.5)
save(fig,'comparison_routes')

configs=['M01','C01','R01','L01'];names=['MLP','CNN','RNN','LSTM']
values=[];sem=[]
for c in configs:
    subset=[r for r in data if r['config']==c]
    # Keep all 60 mappings, including the original semantic grouping.
    v=np.array([float(r['macro_f1_mean']) for r in subset]);assert len(v)==60
    values.append(v);sem.append(float(next(r for r in ranks if r['config']==c)['semantic_f1']))
fig,ax=plt.subplots(figsize=(3.15,1.92));fig.subplots_adjust(left=.17,right=.98,top=.85,bottom=.20)
bp=ax.boxplot(values,positions=np.arange(4),widths=.45,showfliers=False,patch_artist=True,
    boxprops={'facecolor':'#e7edf3','linewidth':.7},medianprops={'color':'#31475d','linewidth':1},whiskerprops={'linewidth':.7},capprops={'linewidth':.7})
for i,v in enumerate(values):
    jitter=np.linspace(-.18,.18,len(v))
    ax.scatter(i+jitter[np.argsort(np.argsort(v))],v,s=3.5,color='#6e8296',alpha=.55,linewidths=0,zorder=3)
ax.scatter(np.arange(4),sem,marker='D',s=21,color='#b0442b',zorder=4,label='의미 기반 매핑 (각 1/60위)')
ax.set_xticks(np.arange(4),names);ax.set_ylim(.65,1.025);ax.set_yticks([.7,.8,.9,1.0]);ax.set_ylabel('3상태 Macro F1',fontsize=7)
ax.grid(axis='y',alpha=.22,lw=.5);ax.spines[['top','right']].set_visible(False)
ax.tick_params(labelsize=7);ax.legend(loc='lower left',bbox_to_anchor=(-.04,1.005),fontsize=6.5,frameon=False,handletextpad=.3,borderaxespad=0)
fig.text(.58,.025,'점: 매핑별 3 seed 평균 · 상자: 사분위수',ha='center',fontsize=6.2)
save(fig,'mapping_sensitivity')
record={'source':source.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model_order':names,'configs':configs,'all_60_mapping_macro_f1_means':[v.tolist() for v in values],'semantic_macro_f1_means':sem,'mapping_ranks':ranks,'new_neural_training':False,'new_linear_fits':2,'test_evaluated':False,'figure_1':'Implemented experiment routes, not a deployed phone application','figure_2':'Descriptive enumeration, not independent hypothesis tests; group class counts may differ'}
(out/'FIGURE_DATA.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Created two v9 figures from measured controls.')
