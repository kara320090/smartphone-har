# ③ 1D CNN 개인 기여 원고 — TensorFlow 정식 검증 결과

**담당 범위:** 1D CNN 모델 구현·학습·검증·성능 분석 및 논문용 결과 정리  
**실험 상태:** TensorFlow 정식 3 seed 학습 및 저장 모델 독립 프로세스 재검증 완료. **공식 test 미평가.**

## 1. 연구 목적

UCI HAR 스마트폰 관성센서 시계열을 이용해 여섯 가지 일상 행동을 분류하는 1차원 합성곱 신경망(1D CNN)을 구현하였다. CNN은 시점 축을 따라 합성곱을 적용하여 인접 시점의 국소적 변화 정보를 학습할 수 있다. 팀 공통 분할·평가지표에 따라 다른 모델과 비교할 수 있도록 실험 구성을 고정하였다.

## 2. 데이터와 전처리

입력은 표본당 **128 시점 × 9채널**의 시계열이다. 센서 채널은 body acceleration, body gyroscope, total acceleration 각각의 3축으로 구성된다. UCI HAR의 제공된 전처리 캐시와 훈련 데이터 기반 정규화 통계를 사용하였다. 훈련 **5,577개**, 검증 **1,775개** 표본은 피험자 집단이 서로 겹치지 않는 고정 분할 S0으로 구성하였다. 공식 test 데이터는 모델 선택과 보고된 결과 산출에 사용하지 않았다.

## 3. 모델 구조

`Conv1D(64, kernel=5, ReLU)` → `MaxPool1D(2)` → `Conv1D(128, kernel=3, ReLU)` → `MaxPool1D(2)` → `Conv1D(128, kernel=3, ReLU)` → `GlobalAveragePooling1D` → `Dense(64, ReLU)` → `Dropout(0.3)` → `Dense(6, linear logits)`로 구성하였다. 모든 합성곱의 패딩은 `same`이며, 시간축 길이는 128 → 64 → 32로 감소한다. 학습 파라미터 수는 **85,574개**이다. 마지막 층은 6개 클래스의 logits을 출력하며, 손실은 `SparseCategoricalCrossentropy(from_logits=True)`를 사용한다.

## 4. 학습 및 검증 조건

TensorFlow 백엔드로 seed **2026, 2027, 2028**에서 각각 모델을 학습하였다. Optimizer는 Adam(learning rate **0.001**), batch size는 **64**, 최대 학습 epoch는 **40**이며, `val_loss` 기준 patience **6**의 조기 종료를 적용하였다. 학습 결과의 `.keras` 모델을 저장한 후 별도 프로세스에서 검증 입력으로 재평가하는 절차를 통과하였다. 세 seed는 동일한 고정 피험자 분할을 사용한다.

## 5. 정식 TensorFlow Validation 결과

| Seed | Validation Accuracy | Validation Macro F1 | 종료 Epoch | 최소 val_loss Epoch |
|---|---:|---:|---:|---:|
| 2026 | 0.8755 | 0.8742 | 8 | 2 |
| 2027 | 0.8969 | 0.8972 | 7 | 1 |
| 2028 | 0.8941 | 0.8952 | 10 | 4 |
| **평균 ± seed 간 표본 SD** | **0.8888 ± 0.0116** | **0.8889 ± 0.0127** | — | — |

검증 정확도의 평균은 **88.88%**, Macro F1 평균은 **0.8889**로 나타났다. seed 간 변동은 고정된 분할에서 초기화 차이에 따른 변동을 나타내며, 미지의 피험자 모집단에 대한 통계적 신뢰구간으로 해석할 수 없다. 검증 손실 최소 시점이 1~4 epoch로 비교적 이르므로 학습 곡선을 통해 조기 과적합 여부를 검토할 필요가 있다. 다만 현 자료만으로 과적합을 단정하지는 않는다.

## 6. 재현성과 결과 해석의 한계

`configs/C01.json`, `scripts/run_cnn_suite.py`, `scripts/cnn_summarize.py`, `scripts/cnn_publish.py`에 실험 절차를 기록하였다. 실제 학습 모델과 예측·검증 기록은 `artifacts/cnn_tensorflow/C01_seedXXXX/`, 집계 결과는 `reports/cnn/`, 파일 해시 목록은 `artifacts/cnn_tensorflow/manifest.json`에 보관한다. 검증용 외부 프로세스 재평가를 확인했으나 정식 **test** 성능은 아직 측정하지 않았다. CNN과 다른 모델의 성능 차이는 파라미터 수, 규제 조건 등이 동일하다는 전제가 없으므로 모델 구조의 순수한 인과 효과로 해석하지 않는다.

## 7. 기여 범위 및 참고문헌

CNN 모델 설계, TensorFlow 실행 코드, 고정 seed 반복 학습, 예측 재검증, 검증 지표 집계 및 논문 결과 정리를 담당하였다. 데이터 수집·전처리 및 팀 공통 평가 설계는 해당 담당자의 작업을 재사용하였다. 수치의 원본은 `docs/team-paper/contributions/03_cnn/final_tensorflow_results.md` 및 `reports/cnn/cnn_summary.json`을 따른다. 이전 Keras/PyTorch 예비 결과와 TensorFlow 정식 결과를 혼합하지 않는다.

- Reyes-Ortiz et al., *Human Activity Recognition Using Smartphones*, UCI Machine Learning Repository, DOI: 10.24432/C54S4K.
- Keras API, `Conv1D`, `GlobalAveragePooling1D`, `EarlyStopping` 문서.
