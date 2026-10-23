"""Recreate figure 2 from the frozen repository data (requires numpy, matplotlib)."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import fontManager
R=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent.parent
REP=R/'reports/confirmatory_20261019'
with (REP/'summary.csv').open(encoding='utf-8') as f: summary=list(csv.DictReader(f))
with (REP/'benchmark_runs.csv').open(encoding='utf-8') as f: bench=list(csv.DictReader(f))
assert json.loads((REP/'AUDIT.json').read_text())['passed']
lookup={(r['condition'],r['level']):r for r in summary}
order=['LR3','FMLP3','MLP3','CNN3','RNN3','LSTM3']
data=[]
for c in order:
    r=lookup[c,'three']
    times=[float(b['window_to_output_median_ms']) for b in bench if b['condition']==c]
    assert abs(np.median(times)-float(r['window_ms_median_across_runs']))<1e-12
    data.append(dict(model=c,f1=float(r['f1_mean']),sd=float(r['f1_seed_sd'] or 0),
                     runs=int(r['runs']),parameters=int(r['parameters']),
                     ms=float(r['window_ms_median_across_runs']),min_ms=min(times),max_ms=max(times)))
ratio=data[1]['ms']/data[0]['ms']
fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':8,'axes.unicode_minus':False,
    'svg.fonttype':'none','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})

# At the final 80 mm width, all model names and primary values are 8 pt.
fig=plt.figure(figsize=(80/25.4,51.4/25.4),facecolor='white')
fig.text(.01,.966,'직접 3상태 · 동일 PC / batch 1 · 전처리 포함',fontsize=7.1,va='top',color='#40464d')
fig.text(.01,.882,'모델',fontsize=8,weight='bold',va='center')
fig.text(.285,.882,'Macro F1 ↑',fontsize=7.3,weight='bold',va='center',ha='center')
fig.text(.70,.882,'CPU 시간(ms) ↓',fontsize=8,weight='bold',va='center',ha='center')
fig.add_artist(plt.Line2D([.01,.995],[.852,.852],transform=fig.transFigure,color='#5f6a73',lw=.6))

bottom,height=.31,.52
ax=fig.add_axes([.43,bottom,.43,height],facecolor='none')
ax.set(xlim=(0,6.2),ylim=(5.5,-.5),yticks=[],xticks=[0,2,4,6])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#6a747d')
ax.spines['bottom'].set_linewidth(.6)
ax.tick_params(axis='x',labelsize=6.9,pad=2,length=2,width=.6)
ax.grid(axis='x',color='#e1e5e8',lw=.45)
ax.set_axisbelow(True)
for i,r in enumerate(data):
    yc=bottom+height*(1-(i+.5)/6)
    if i<2:
        fig.add_artist(Rectangle((.005,yc-.04),.99,.08,transform=fig.transFigure,
            facecolor='#e9f2ef',edgecolor='none',zorder=-2))
    color='#267968' if i<2 else '#7b8996'
    ax.barh(i,r['ms'],height=.46,color=color,zorder=2)
    ax.errorbar(r['ms'],i,xerr=[[r['ms']-r['min_ms']],[r['max_ms']-r['ms']]],
        fmt='none',ecolor='#263d4d',elinewidth=.7,capsize=2,capthick=.7,zorder=3)
    fig.text(.01,yc,r['model'],fontsize=8,va='center',weight='bold' if i<2 else 'normal')
    fig.text(.285,yc,f"{r['f1']:.4f}",fontsize=8,va='center',ha='center')
    fig.text(.995,yc,f"{r['ms']:.3f}",fontsize=8,va='center',ha='right')
fig.add_artist(plt.Line2D([.01,.995],[.65,.65],transform=fig.transFigure,color='#b9c8c3',lw=.5))
fig.text(.01,.205,'F1: NN 3 seed 평균, LR 단일 학습. SD는 표 1.',fontsize=6.3,va='center',color='#40464d')
fig.text(.01,.153,'막대: 실행 중앙값의 중앙값 / 선: 실행별 중앙값 범위.',fontsize=6.1,va='center',color='#40464d')
fig.add_artist(Rectangle((.005,.012),.99,.093,transform=fig.transFigure,facecolor='#e9f2ef',edgecolor='none',zorder=-2))
fig.text(.5,.059,f'같은 18특징: FMLP3 / LR3 처리 시간 {ratio:.1f}배',
         fontsize=7.1,weight='bold',ha='center',va='center',color='#185c4f')
for ext in ('png','svg'):
    fig.savefig(OUT/'figures'/f'accuracy_cost.{ext}',dpi=600)
plt.close(fig)

