# 검증 피험자 구성 민감도 진단

기존 M01·C01·R01·L01의 세 seed, 12회 저장 예측으로 계산했다. 전체 5명·1,775구간은 대표 결과로 유지한다. 검증 피험자 1·6·14·21·23을 각각 한 명씩 제외해 나머지의 6클래스 Macro F1을 다시 계산했다.

이는 재학습 없는 사후 진단이며 LOSO 교차검증, 새로운 분할 학습, 독립 test 또는 저성능 피험자 제거에 의한 성능 향상이 아니다. 제외 시 표본 수와 클래스 구성도 함께 달라지므로 피험자 한 명의 인과 효과로 해석하지 않는다. 다섯 제외 경우를 모두 보고한다.

`runs.csv`: 4구성×3seed×(전체+5제외)의 72개 지표. `summary.csv`: 24개 조건의 seed 평균·표본 SD. `paired_seed_gaps.csv`: CNN-LSTM 및 LSTM-MLP의 같은 seed 차이. `analysis.json`: 원본 예측 해시·지표 대조·해석 범위.

```powershell
python tools/analyze_subject_composition.py --lstm-runs runs/lstm --output reports/subject_composition
```

LSTM 전체 예측은 기존 LSTM Release의 실행 폴더에서 준비한다. scikit-learn이 설치되어 있으면 72개 계산을 독립 지표 구현과 대조한다. 기본 계산은 Python 표준 라이브러리만 사용한다.
