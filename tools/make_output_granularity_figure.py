"""Show actual saved predictions and the information retained by output grouping."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs/application-evidence-20261015"
with (ROOT/"reports/output_granularity/l02_error_examples.csv").open(encoding="utf-8") as stream:
    rows=[r for r in csv.DictReader(stream) if r["seed"]=="2026"]
specs=[
    ("이동 상태 요약",lambda r:r["true_label"] in ["0","1","2"] and r["predicted_label"] in ["0","1","2"]),
    ("자세의 큰 범주 요약",lambda r:r["true_label"] in ["3","4"] and r["predicted_label"] in ["3","4"]),
    ("묶어도 남는 오류",lambda r:r["three_group_error"]=="True"),
]
selected=[{"example_title":title,**next(r for r in rows if predicate(r))} for title,predicate in specs]
fine=["걷기","계단 오르기","계단 내리기","앉기","서기","눕기"]
coarse=["이동","앉거나 서 있음","누워 있음"]
font_path=Path("C:/Windows/Fonts/malgun.ttf")
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({"font.family":font_manager.FontProperties(fname=str(font_path)).get_name(),"font.size":11,"svg.fonttype":"path","axes.unicode_minus":False})
fig=plt.figure(figsize=(11,4.8),facecolor="white")
fig.suptitle("활동을 얼마나 자세히 구분해야 하는가?",x=.055,y=.96,ha="left",fontsize=17,weight="bold",color="#193444")
fig.text(.055,.865,"기존 6종 예측을 목적에 맞는 3개 상태로 변환한 실제 사례",fontsize=11,color="#536A78")
for i,sample in enumerate(selected):
    ax=fig.add_axes([.045+i*.315,.205,.295,.59]);ax.set_axis_off()
    failed=sample["three_group_error"]=="True"
    color="#B34D40" if failed else "#167B81"
    ax.add_patch(FancyBboxPatch((.01,.01),.98,.98,boxstyle="round,pad=0.005,rounding_size=.035",facecolor="#F6F8FA",edgecolor="#D8E1E6",transform=ax.transAxes))
    ax.text(.075,.88,sample["example_title"],fontsize=13,weight="bold",color=color)
    ax.text(.075,.69,"정답",fontsize=9,color="#5A6A76")
    ax.text(.075,.59,fine[int(sample["true_label"])],fontsize=14,weight="bold",color="#243E4D")
    ax.text(.075,.44,"6종 예측: "+fine[int(sample["predicted_label"])],fontsize=11,color="#394D5A")
    ax.text(.075,.29,"3상태 표시: "+coarse[int(sample["predicted_three_group"])],fontsize=11,weight="bold",color=color)
    ax.text(.075,.13,"상태 요약도 틀림" if failed else "세부 동작은 틀렸지만 상태 요약은 맞음",fontsize=9,color=color)
    ax.text(.075,.055,f'ID {sample["sample_id"]} / 피험자 {sample["subject"]}',fontsize=8,color="#617681")
fig.text(.055,.10,"L02 seed 2026 저장 예측에서 유형별 첫 표본. 유리한 사례만 보여주지 않도록 상태 요약 실패도 포함.",fontsize=9,color="#536A78")
fig.text(.055,.048,"동일 개발 validation의 사후 분석. 구분 정보를 줄인 출력이며, 모델 성능 향상·실제 앱 검증을 뜻하지 않음.",fontsize=9,color="#536A78")
(OUT/"figures").mkdir(parents=True,exist_ok=True)
for ext in ("png","svg"):
    path=OUT/"figures"/("activity_summary_examples."+ext)
    fig.savefig(path,dpi=220,facecolor="white")
    if ext=="svg":path.write_text("\n".join(s.rstrip() for s in path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
(OUT/"FIGURE_EXAMPLES.json").write_text(json.dumps({"selection":"First sample in original order for each example type; L02 seed 2026 fixed, no best-run selection", "examples":selected},ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
plt.close(fig)
print("Rendered three actual prediction examples.")
