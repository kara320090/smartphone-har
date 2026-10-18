"""Generate scientific figures only from the frozen test/benchmark result files."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import fontManager,FontProperties
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reports/confirmatory_20261019'
OUT=ROOT/'docs/dcs-team-paper-research-v10/figures'
OUT.mkdir(parents=True,exist_ok=True)
fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
                     'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
                     'savefig.facecolor':'white'})
assert json.loads((REP/'AUDIT.json').read_text())['passed']
with (REP/'summary.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
with (REP/'benchmark_runs.csv').open(encoding='utf-8') as f:bench=list(csv.DictReader(f))
s={(r['condition'],r['level']):r for r in rows}

fig,ax=plt.subplots(figsize=(4.0,1.7))
ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
def box(x,y,w,h,label,color='#edf3f8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.1',
                              facecolor=color,edgecolor='#506172',linewidth=.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
box(.1,2.55,3.0,1.1,'학습 16명\n5,577구간')
box(3.5,2.55,3.0,1.1,'검증 5명\n1,775구간')
box(7.0,2.55,2.9,1.1,'고정 → test 9명\n2,947구간','#dfeee7')
box(.1,.35,4.55,1.4,'통계 특징 18개\nLR3 ↔ FMLP3')
box(5.0,.35,4.9,1.4,'시계열 128×9\n6종 병합 ↔ 직접 3상태')
for a,b in [((3.14,3.1),(3.45,3.1)),((6.55,3.1),(6.95,3.1))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#37495c','lw':1.2})
ax.text(5,2.15,'학습 통계만 적용 · 조건·모델 고정 후 test',ha='center',fontsize=9)
fig.tight_layout(pad=.2)
for ext in ('png','svg'):fig.savefig(OUT/f'evaluation_design.{ext}',dpi=360,bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(4.5,3.0),sharey=True)
labels={'LR3':'LR3','FMLP3':'FMLP3','MLP3':'MLP3','CNN3':'CNN3','RNN3':'RNN3','LSTM3':'LSTM3'}
colors={'LR3':'#c33a32','FMLP3':'#d48231','MLP3':'#466b8c','CNN3':'#2a887a','RNN3':'#8870a4','LSTM3':'#3452a4'}
for i,c in enumerate(labels):
    r=s[c,'three'];x=float(r['window_ms_median_across_runs']);y=float(r['f1_mean']);sd=float(r['f1_seed_sd'] or 0)
    timings=[float(b['window_to_output_median_ms']) for b in bench if b['condition']==c]
    axes[0].errorbar(y,i,xerr=sd,fmt='o',markersize=6,capsize=3,color=colors[c],zorder=3)
    axes[1].errorbar(x,i,xerr=[[x-min(timings)],[max(timings)-x]],fmt='o',markersize=6,capsize=3,color=colors[c],zorder=3)
axes[0].set_yticks(range(6),labels.values())
axes[0].set_xlim(.98,1.0015);axes[0].set_xticks([.98,.99,1.0])
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.2f}'))
axes[0].set_title('test Macro F1',fontsize=12,pad=10)
axes[1].set_xscale('log');axes[1].set_xlim(.04,12);axes[1].set_xticks([.1,1,10])
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
axes[1].xaxis.set_minor_formatter(NullFormatter())
axes[1].set_title('CPU ms (로그)',fontsize=12,pad=10)
for ax in axes:
    ax.set_ylim(5.6,-.6);ax.grid(axis='x',alpha=.25,which='major');ax.tick_params(labelsize=11)
fig.subplots_adjust(left=.21,right=.98,bottom=.23,top=.85,wspace=.30)
fig.text(.02,.04,'성능: seed 평균±SD / 시간: 중앙값과 실행별 범위\nLR은 단일 학습 · 시간은 전처리 포함',fontsize=10,color='#40454b')
for ext in ('png','svg'):fig.savefig(OUT/f'accuracy_cost.{ext}',dpi=360,bbox_inches='tight')
plt.close(fig)
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (REP/'summary.csv',REP/'benchmark_runs.csv',REP/'AUDIT.json',REP/'FINAL_LOCK.json')}
(OUT/'FIGURE_DATA.json').write_text(json.dumps({'summary':rows,'source_sha256':sources},ensure_ascii=False,indent=2),encoding='utf-8')
print('Two verified-result figures written')
