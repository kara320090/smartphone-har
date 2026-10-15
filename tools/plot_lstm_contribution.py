"""Render existing LSTM contribution evidence; requires audit output, no training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/research-contribution-review-20261015"
DATA = json.loads((FOLDER / "EVIDENCE.json").read_text(encoding="utf-8"))
assert DATA["passed"]
font = Path("C:/Windows/Fonts/malgun.ttf")
if font.is_file():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "svg.fonttype": "path"})
colors = {"L01": "#536B80", "L02": "#087E8B", "L03": "#C16649"}
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), gridspec_kw={"width_ratios": [1, 1.1]})
fig.subplots_adjust(left=.075, right=.975, top=.82, bottom=.22, wspace=.34)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E9ED", linewidth=.7)
    ax.set_axisbelow(True)

ax = axes[0]
for x, condition in enumerate(("L01", "L02", "L03")):
    summary = DATA["conditions"][condition]
    values = [r["macro_f1"] for r in DATA["runs_recomputed_from_confusion"] if r["condition"] == condition]
    ax.scatter([x-.12, x, x+.12], values, s=42, color=colors[condition], zorder=4)
    ax.errorbar(x, summary["macro_f1_mean"], yerr=summary["macro_f1_seed_sample_sd"], fmt="D", color="#202B33", ms=4, capsize=4, linewidth=1.2, zorder=5)
    ax.text(x, .928, f'{summary["macro_f1_mean"]:.4f}', ha="center", weight="bold", fontsize=10)
ax.set_xticks([0,1,2], ["L01 · 64유닛\n21,222개", "L02 · 32유닛\n6,630개", "L03 · Dropout\n21,222개"])
ax.set_xlim(-.6, 2.6)
ax.set_ylim(.79, .945)
ax.set_ylabel("개발 Macro F1")
ax.set_title("(a) 기존 9회 실행: 규모와 평균 성능", loc="left", fontsize=11, pad=16)

ax = axes[1]
ax.axhline(0, color="#252F37", linewidth=1)
for index, condition in enumerate(("L02", "L03")):
    rows = DATA["paired_seed_differences"][condition + "_minus_L01"]["per_seed"]
    xpos = [x + (-.14 if index == 0 else .14) for x in range(3)]
    vals = [r["delta_macro_f1"] for r in rows]
    ax.scatter(xpos, vals, color=colors[condition], s=52, label=condition + " - L01", marker="o" if index==0 else "s", zorder=4)
    for x, y in zip(xpos, vals):
        ax.vlines(x, 0, y, color=colors[condition], linewidth=2, alpha=.65)
        dy = 9 if y >= 0 else (-29 if index == 0 else -15)
        ax.annotate(f"{y:+.4f}", (x,y), xytext=(-8 if index == 0 else 8, dy), textcoords="offset points", ha="right" if index == 0 else "left", color=colors[condition], fontsize=9)
ax.set_xticks([0,1,2], ["2026", "2027", "2028"])
ax.set_xlabel("학습 seed")
ax.set_ylabel("L01 대비 Macro F1 차이")
ax.set_ylim(-.10, .052)
ax.set_xlim(-.68, 2.85)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.set_title("(b) L02는 2/3회 개선, 1/3회 하락", loc="left", fontsize=11, pad=16)
fig.suptitle("소형 LSTM은 유망한 후보지만, 성능 이점은 모든 반복에서 일치하지 않았다", fontsize=13, weight="bold", y=.97)
fig.text(.075, .065, "동일 개발 사람 분할 · 각 조건 3 seed · 공식 test 미평가\n점: 개별 실행 / ◆: 평균 / 세로 오차선: seed 표본 SD. L01·L02의 초기 가중치가 같다는 의미는 아님.", fontsize=9, color="#465664")
(FOLDER / "figures").mkdir(exist_ok=True)
for extension in ("png", "svg"):
    fig.savefig(FOLDER / "figures" / ("lstm_capacity_evidence." + extension), dpi=240, facecolor="white")
svg_path = FOLDER / "figures/lstm_capacity_evidence.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
plt.close(fig)
print("Rendered existing evidence in PNG and SVG.")
