"""Publication figures from the audited output-granularity results (no training)."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT/'docs/dcs-team-paper-research-v8/figures')
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
font = Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams.update({'font.size': 7, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
navy, teal, orange, gray = '#18334D', '#007F82', '#C35D22', '#CED9E2'
source = ROOT/'reports/output_granularity/summary.csv'
with source.open(encoding='utf-8') as f:
    data = {(r['condition'], r['level']): r for r in csv.DictReader(f)}

def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=450, facecolor='white')
    fig.savefig(OUT/(name+'.svg'), facecolor='white')
    svg = OUT/(name+'.svg')
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')
    plt.close(fig)

fig = plt.figure(figsize=(80/25.4, 61/25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
def box(x, y, w, h, label, color, fill, size=7, weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.015',linewidth=.8,edgecolor=color,facecolor=fill))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=color,fontweight=weight,linespacing=1.5)
box(.025,.84,.95,.135,'스마트폰 가속도·자이로 신호\n50 Hz · 128시점 × 9채널',navy,'#F0F4F7',7.3,'bold')
ax.annotate('',xy=(.5,.79),xytext=(.5,.83),arrowprops={'arrowstyle':'->','lw':1,'color':navy})
box(.025,.67,.95,.115,'6종 행동 분류: MLP · CNN · RNN · LSTM',navy,'#FFFFFF',7.1)
ax.text(.5,.624,'정답과 예측에 같은 고정 매핑 적용',ha='center',va='center',fontsize=6.6,color='#465B6D')
mapping = [('걷기 · 계단 오르기 · 내리기','이동'),('앉기 · 서기','앉거나 서 있음'),('눕기','누워 있음')]
for i,(fine,coarse) in enumerate(mapping):
    y=.47-i*.137
    box(.025,y,.555,.107,fine,navy,'#F5F7F9',6.8)
    ax.annotate('',xy=(.64,y+.0535),xytext=(.59,y+.0535),arrowprops={'arrowstyle':'->','lw':.9,'color':teal})
    box(.655,y,.32,.107,coarse,teal,'#E7F4F3',6.8,'bold')
ax.text(.5,.13,'활용 제안: 구간별 활동 상태 표시',ha='center',va='center',fontsize=7.3,fontweight='bold',color=navy)
ax.text(.5,.052,'검증 범위: 저장 구간의 분류·변환 (앱·일별 시간 기록 미검증)',ha='center',va='center',fontsize=5.9,color='#536575')
save(fig,'sensor_to_activity_summary')

conditions=['M01','C01','R01','L01']
names=['MLP','CNN','RNN','LSTM']
totals=[int(data[(c,'three')]['fine_error_events_across_seeds']) for c in conditions]
remaining=[int(data[(c,'three')]['coarse_error_events_across_seeds']) for c in conditions]
within=[a-b for a,b in zip(totals,remaining)]
fig,ax=plt.subplots(figsize=(80/25.4, 53/25.4))
fig.subplots_adjust(left=.145,right=.985,top=.80,bottom=.23)
y=range(4)
ax.barh(y,within,height=.48,color=gray,label='동일 상태 내 혼동')
ax.barh(y,remaining,left=within,height=.48,color=orange,label='상태 간 혼동')
for i,(a,b) in enumerate(zip(totals,remaining)):
    ax.text(a+28,i,f'{b} / {a}',va='center',fontsize=6.8,color=navy)
ax.set_yticks(list(y),names,fontsize=7.2)
ax.invert_yaxis();ax.set_xlim(0,1810);ax.set_xticks([0,500,1000,1500])
ax.tick_params(axis='both',length=0,pad=3,labelsize=6.6)
ax.set_xlabel('6종 분류 오류 사건 수 (세 seed 합계)',fontsize=6.8,labelpad=5)
ax.grid(axis='x',color='#E7ECF0',linewidth=.6);ax.set_axisbelow(True)
for s in ax.spines.values():s.set_visible(False)
fig.legend(loc='upper left',bbox_to_anchor=(.07,1.02),ncol=2,frameon=False,fontsize=6.5,handlelength=1.5,columnspacing=1.0)
fig.text(.50,.855,'막대 옆 수치: 상태 간 오류 / 전체 6종 오류',ha='center',fontsize=6.3,color=navy)
fig.text(.5,.038,'같은 1,775개 구간을 반복 평가한 오류 사건이며 독립 표본 수가 아님',ha='center',fontsize=5.7,color='#536575')
save(fig,'error_composition')

record={'date_korea':'2026-10-19','source':'reports/output_granularity/summary.csv','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'conditions':conditions,'fine_error_events':totals,'within_state_error_events':within,'between_state_error_events':remaining,'mapping_six_to_three':[0,0,0,1,1,2],'method':'Map each six-class argmax label and ground-truth label identically; no probability aggregation or retraining.','post_hoc':True,'independent_test':False,'new_training_performed':False,'application_implemented':False,'images':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['sensor_to_activity_summary.png','error_composition.png']}}
(OUT/'FIGURE_DATA.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record,ensure_ascii=True))
