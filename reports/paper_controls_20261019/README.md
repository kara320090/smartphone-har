# v9 후속 대조 실험

2026-10-19. [프로토콜](../../docs/paper-controls-20261019/PROTOCOL_KO.md)은 실행 전 커밋 `3216845`에 고정했다. 신경망 결과를 이미 확인한 뒤 설계한 추가 분석이며 사전등록 연구가 아니다. 공식 test는 로딩·평가하지 않았다.

```powershell
python tools/paper_controls.py --processed-data C:/Users/soma/Desktop/processed_data.npz --output reports/paper_controls_reproduction
python tools/paper_controls.py --processed-data C:/Users/soma/Desktop/processed_data.npz --output reports/paper_controls_reproduction --verify-only
```

기존 결과를 덮어쓰지 않도록 새 출력 폴더를 지정한다. 입력 NPZ와 팀 표준화 통계는 해시를 검사한다. 실제 실행 환경은 Python·NumPy·scikit-learn 버전과 함께 `execution_start.json`에 기록했다(scikit-learn 1.7.2).

| 파일 | 내용 |
| --- | --- |
| execution_start.json | 시작 시각, 고정 조건, 데이터·통계·프로토콜·코드 해시, 환경 |
| linear_summary.csv / linear_subjects.csv | 전체·사람별 F1, 정확도, 오류 수 |
| LR6_model.json / LR3_model.json | 학습된 계수·편향, 특징 표준화값, 조건·수렴 반복 수 |
| LR6_predictions.csv / LR3_predictions.csv | 검증 1,775개의 ID·사람·정답·예측·확률 |
| LR*_metrics.json | 클래스별 지표와 혼동행렬 |
| LR*_warnings.json | 수렴을 포함한 경고 기록, 두 실행 모두 빈 목록 |
| all_mapping_runs.csv | 60개 매핑 × 12회 신경망 예측 = 720건 |
| all_mapping_summary.csv / mapping_rank_summary.csv | 모델별 세 seed 평균·표본 SD와 매핑 순위 |
| verification.json | 독립 지표 계산 대조, 예측 ID·정답·사람 정렬 검사, 사용한 12개 예측 파일 해시 |
| reload_verification.json | 새 프로세스의 전체 validation 재예측 일치, 최대 확률 오차 0.0 |

LR6: 6종 F1 0.7741516886, 매핑 후 3상태 F1 0.9985485639. LR3: 직접 3상태 F1 1.0, 이 검증 분할에서 0/1,775 오류. LR6의 3상태 오류 2개는 사람 14에서 관찰했다. LR3는 다섯 검증 사람 모두 이번 표본에서는 무오류다. 반복 실험·독립 test 결과로 확대하지 않는다.

LR6/3는 각각 단일 결정론적 최적화 실행이다. 고정한 lbfgs 조건에서 seed 반복 SD를 만들지 않았다. 114/57은 다항 로지스틱 회귀의 저장 계수·편향 수다. 특징 표준화 상수 36개와 팀 시계열 표준화 상수 18개는 이 수에 포함하지 않았다. 전체 비용·실기기 성능은 미측정이다.

Macro F1은 혼동행렬로 직접 계산한 값과 scikit-learn 값을 대조했다. 모델 재로딩에는 JSON의 계수와 NumPy softmax만 사용했다. 매핑 열의 0/1/2는 각각 크기 3/2/1의 그룹이며, 의미 기반 매핑은 `0,0,0,1,1,2`다. 원래 과제의 argmax 예측을 묶었고 확률 합산으로 새로운 argmax를 선택하지 않았다.

이 분석은 단일 개발 검증의 탐색적 결과다. 매핑 간 클래스별 표본 수가 달라 60개 결과를 독립 확률 표본이나 p값으로 해석하지 않는다. 선형 기준·신경망의 입력 표현과 학습 절차 차이도 함께 고려해야 한다.
