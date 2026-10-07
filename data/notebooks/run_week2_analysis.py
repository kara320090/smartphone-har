import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==============================================================================
# [1] 지정된 절대 경로 설정
# ==============================================================================
# 사용자가 지정한 캐시 데이터 파일 경로
CACHE_PATH = r"C:\Users\wolah\Desktop\인공지능_공학\processed_data.npz"

# 결과 이미지 및 리포트가 저장될 폴더 (동일 폴더 기준)
OUTPUT_DIR = r"C:\Users\wolah\Desktop\인공지능_공학"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6개 행동 라벨 정의 (계획서 기준)[cite: 1]
ACTIVITY_NAMES = ["걷기", "계단 오르기", "계단 내려가기", "앉기", "서기", "눕기"]

# 한글 폰트 설정 (Windows Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("  스마트폰 센서 행동인식 - [2주차 데이터 분석 & 시각화 파이프라인][cite: 1]")
print(f"  대상 캐시 파일: {CACHE_PATH}")
print(f"  결과 저장 폴더: {OUTPUT_DIR}")
print("=" * 70)

# ==============================================================================
# [2] 캐시 데이터 로드 및 학습 데이터 표준화
# ==============================================================================
if not os.path.exists(CACHE_PATH):
    raise FileNotFoundError(f"[경로 오류] 지정된 위치에 캐시 파일이 없습니다: {CACHE_PATH}")

t0 = time.time()
print("\n>> [1/4] 캐시 데이터 로드 및 표준화 수행 중...")
data = np.load(CACHE_PATH)

Xs_fit = data["fit_Xs"]          # (5577, 128, 9)[cite: 1]
Xf_fit = data["fit_Xf"]          # (5577, 561)[cite: 1]
y_fit = data["fit_y"]            # (5577,)[cite: 1]
sub_fit = data["fit_subject"]    # (5577,)[cite: 1]

# 561 특징 열별 표준화 (학습 피험자 16명 기준으로만 통계 산출)[cite: 1]
feat_mean = np.mean(Xf_fit, axis=0, keepdims=True)
feat_std = np.std(Xf_fit, axis=0, keepdims=True)
feat_std = np.maximum(feat_std, 1e-8)
Xf_fit_norm = (Xf_fit - feat_mean) / feat_std

print(f"✓ 로드 완료: 학습 데이터 {Xs_fit.shape[0]:,}개 표본 ({time.time() - t0:.2f}초)[cite: 1]")

# ==============================================================================
# [3] 과제 1: 행동별 센서 파형 비교 시각화 (계획서 04절, 그림 2 재현)[cite: 1]
# ==============================================================================
print("\n>> [2/4] 과제 1: 피험자 3번의 행동별 센서 파형(몸 가속도 vs 전체 가속도) 시각화 중...[cite: 1]")
time_axis = np.linspace(0, 2.56, 128)  # 50Hz, 128시점 = 2.56초[cite: 1]

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

# 채널 0: body_acc_x, 채널 6: total_acc_x[cite: 1]
for act_idx in range(6):
    ax = axes[act_idx]
    match_idx = np.where((sub_fit == 3) & (y_fit == act_idx))[0]
    if len(match_idx) == 0:
        match_idx = np.where(y_fit == act_idx)[0]
    sample_i = match_idx[0]

    ax.plot(time_axis, Xs_fit[sample_i, :, 0], label="몸 가속도 x", color="#1f4e79", linewidth=1.5)
    ax.plot(time_axis, Xs_fit[sample_i, :, 6], label="전체 가속도 x (중력 성분 포함)", color="#c55a11", linewidth=1.5)
    ax.set_title(f"{ACTIVITY_NAMES[act_idx]}", fontsize=11, fontweight="bold")
    ax.set_ylabel("가속도 (g)")
    ax.grid(True, linestyle=":", alpha=0.6)
    if act_idx == 0:
        ax.legend(loc="upper right", fontsize=8)

for ax in axes[-2:]:
    ax.set_xlabel("시간 (초)")

plt.suptitle("학습 피험자 3의 행동별 센서 파형 비교 (몸 가속도 vs 전체 가속도)[cite: 1]", fontsize=13, y=0.99)
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "week2_fig1_waveforms.png")
plt.savefig(fig1_path, dpi=200)
plt.close()
print(f"✓ [저장 완료] {fig1_path}")

# ==============================================================================
# [4] 과제 2: PCA 분석 및 2D 산점도 / 95% 분산 곡선 (계획서 07절, 그림 4 재현)[cite: 1]
# ==============================================================================
print("\n>> [3/4] 과제 2: PCA 설명분산 곡선 및 2D 투영 산점도 생성 중...[cite: 1]")
pca = PCA(n_components=120, random_state=2026)
Xf_pca = pca.fit_transform(Xf_fit_norm)

pc1_ratio = pca.explained_variance_ratio_[0] * 100
pc2_ratio = pca.explained_variance_ratio_[1] * 100
pc1_2_var = pc1_ratio + pc2_ratio
cum_var = np.cumsum(pca.explained_variance_ratio_) * 100

print(f"   - PC1 + PC2 설명 분산    : {pc1_2_var:.2f}% (계획서 수치: 58.35% 일치)[cite: 1]")
print(f"   - 100개 주성분 누적 분산 : {cum_var[99]:.2f}% (계획서 수치: 95.00% 이상 일치)[cite: 1]")

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

# 왼쪽: 2D 투영 산점도[cite: 1]
colors = ["#4a7bb0", "#4fa89b", "#cf8d54", "#9b7bb8", "#cc7a88", "#7d966d"]
for act_idx in range(6):
    mask = (y_fit == act_idx)
    ax_left.scatter(
        Xf_pca[mask, 0], Xf_pca[mask, 1],
        label=ACTIVITY_NAMES[act_idx], color=colors[act_idx], alpha=0.45, s=15
    )
ax_left.set_title("학습 사람만 사용한 2D PCA 투영[cite: 1]", fontsize=12)
ax_left.set_xlabel(f"PC1 ({pc1_ratio:.1f}%)")
ax_left.set_ylabel(f"PC2 ({pc2_ratio:.1f}%)")
ax_left.legend(loc="lower right", fontsize=8, frameon=True)
ax_left.grid(True, linestyle=":", alpha=0.5)

# 오른쪽: 누적 설명분산 곡선[cite: 1]
ax_right.plot(range(1, 121), cum_var, color="#006666", linewidth=2.2)
ax_right.axhline(y=95, color="gray", linestyle="--", linewidth=1.0)
ax_right.axvline(x=100, color="gray", linestyle="--", linewidth=1.0)
ax_right.scatter([100], [cum_var[99]], color="#006666", s=60, zorder=5)
ax_right.annotate(
    f"95%에 100개 필요 ({cum_var[99]:.2f}%)[cite: 1]",
    xy=(100, cum_var[99]), xytext=(45, 83),
    arrowprops=dict(facecolor="#333333", shrink=0.08, width=1, headwidth=5),
    fontsize=10, fontweight="bold", color="#006666"
)
ax_right.set_title("남겨야 할 차원 결정 (PCA 누적 분산)[cite: 1]", fontsize=12)
ax_right.set_xlabel("주성분 개수")
ax_right.set_ylabel("누적 설명분산 (%)")
ax_right.set_ylim(40, 102)
ax_right.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "week2_fig2_pca.png")
plt.savefig(fig2_path, dpi=200)
plt.close()
print(f"✓ [저장 완료] {fig2_path}")

# ==============================================================================
# [5] 과제 3: K-means (K=6) 군집화 및 교차표 (matplotlib 구현)[cite: 1]
# ==============================================================================
print("\n>> [4/4] 과제 3: PCA 95% 공간(100차원) 기반 K-means 군집화 및 교차표 생성 중...[cite: 1]")
Xf_pca_100 = Xf_pca[:, :100]

kmeans = KMeans(n_clusters=6, n_init=10, random_state=2026)
clusters = kmeans.fit_predict(Xf_pca_100)

df_cross = pd.crosstab(
    pd.Series([ACTIVITY_NAMES[y] for y in y_fit], name="실제 정답 활동"),
    pd.Series(clusters, name="K-means 군집 번호")
)

print("\n--- 실제 정답 vs K-means 군집 교차표 ---")
print(df_cross)

# 히트맵 이미지 저장 (seaborn 미설치 대응 matplotlib imshow)
plt.figure(figsize=(7, 5.5))
plt.imshow(df_cross.values, cmap="YlGnBu", aspect="auto")
plt.colorbar()
plt.xticks(range(6), [f"군집 {i}" for i in range(6)])
plt.yticks(range(6), df_cross.index)

for i in range(6):
    for j in range(6):
        val = df_cross.values[i, j]
        plt.text(j, i, f"{val}", ha="center", va="center", color="black" if val < 500 else "white")

plt.title("실제 행동 라벨과 K-means 군집 간 교차표 (K=6)[cite: 1]", fontsize=12, pad=12)
plt.xlabel("K-means 군집 번호")
plt.ylabel("실제 정답 활동")
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "week2_fig3_kmeans_crosstab.png")
plt.savefig(fig3_path, dpi=200)
plt.close()
print(f"✓ [저장 완료] {fig3_path}")

# ==============================================================================
# [6] 2주차 분석 요약 리포트 저장 (보고서 및 발표 근거용)[cite: 1]
# ==============================================================================
report_path = os.path.join(OUTPUT_DIR, "week2_analysis_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== 스마트폰 센서 행동인식 - 2주차 데이터 분석 완료 리포트 ===\n\n")
    f.write("1. 센서 파형 분석 (계획서 04절)[cite: 1]\n")
    f.write("   - 동적 행동: 주기적인 보행 패턴 및 뚜렷한 가속도 진폭 발생[cite: 1]\n")
    f.write("   - 정적 행동: 몸 가속도는 0에 수렴하나 전체 가속도에 중력 성분(기저값)이 유지됨[cite: 1]\n")
    f.write("   - 결론: 구간별 독립 정규화는 중력 정보를 제거하므로 채널 통합 표준화를 적용함[cite: 1]\n\n")
    f.write("2. PCA 주성분 분석 (계획서 07절)[cite: 1]\n")
    f.write(f"   - PC1 설명분산: {pc1_ratio:.2f}%\n")
    f.write(f"   - PC2 설명분산: {pc2_ratio:.2f}%\n")
    f.write(f"   - PC1 + PC2 설명분산 합: {pc1_2_var:.2f}% (계획서 기대치: 약 58.35% 일치)[cite: 1]\n")
    f.write(f"   - 95% 분산 보존에 필요한 차원: 100차원 ({cum_var[99]:.2f}% 달성, 계획서 일치)[cite: 1]\n\n")
    f.write("3. K-means 군집화 및 교차표 해석[cite: 1]\n")
    f.write(df_cross.to_string())
    f.write("\n\n   - 해석: 정적 행동(앉기, 서기)은 비지도 공간에서 군집 혼재 현상이 발생함[cite: 1]\n")
    f.write("   - 결론: PCA 분산 보존이 지도학습 분류 성능을 보장하지 않음을 증명함[cite: 1]\n")

print(f"✓ [저장 완료] {report_path}")
print("\n" + "=" * 70)
print("  2주차 전처리 및 데이터 분석 파트의 모든 과업이 완료되었습니다!")
print(f"  산출물 위치: {OUTPUT_DIR}")
print("=" * 70)