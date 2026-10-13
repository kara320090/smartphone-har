# 네 기준 모델의 사후 오류 분석

2026-10-13 · M01/C01/R01/L01 × seed 2026/2027/2028 · 같은 개발 validation 1,775개

저장 예측 21,300행의 고유 ID·사람·정답 순서를 검사했다. 전체와 피험자별 지표를 예측에서 다시 계산해 기존 감사 값과 대조했다. 새 학습과 공식 test 평가는 없다.

- `class_runs.csv`, `class_summary.csv`: 실행별 및 seed 요약 클래스 precision·recall·F1
- `confusion_mean.csv`: 실제→예측의 평균 오류 수와 실제 클래스 support로 나눈 조건부 비율
- `directional_error_runs.csv`, `directional_error_summary.csv`: 앉기→서기 및 반대 방향의 오류, 평균과 SD
- `subject_summary.csv`: 피험자 내부 여섯 클래스 Macro F1의 세 seed 평균·SD와 Accuracy
- `behavior_group_summary.csv`: 전체 혼동행렬의 세 동적/세 정적 클래스 F1 평균. 별도 3클래스 모델의 점수가 아님
- `analysis.json`: 입력 근거 해시·검사 범위·해석 한계

[연구 초안 v2와 해석](../../docs/dcs-team-paper-research-v2/README.md) · [계산 코드](../../tools/analyze_team_errors.py)

저장소 루트에서 LSTM Release의 `runs/lstm`을 준비한 뒤 새 출력 경로로 재계산한다. 표준 Python만 필요하다. 다른 담당자의 예측은 저장소에 있다.

```powershell
python tools/analyze_team_errors.py --lstm-runs runs/lstm --output reports/team_error_analysis_recheck
```

세 seed는 같은 검증 구간의 반복 학습 결과이며 독립 사용자 표본이 아니다. SD는 사람 모집단 신뢰구간이 아니다. 모델 용량·헤드·Dropout·탐색 예산이 다르므로 게이트의 인과 효과·스마트폰 속도·통계적 우월성은 판정하지 않는다.
