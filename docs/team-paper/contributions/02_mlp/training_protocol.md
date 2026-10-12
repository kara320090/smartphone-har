# MLP 고정 실험 규약 — 한윤섭

작성일: 2026-10-12 (한국 시간). 정식 학습 전에 작성한 이번 로컬 MLP 실행 규약이다. 팀 전체의 최종 승인이나 최종 test 평가를 뜻하지 않는다.

| 항목 | 고정값 |
|---|---|
| 실행 설정 | `configs/M01.json`, 새 실행 ID `M01_seed2026/2027/2028` |
| 구조 출처 | 기존 `src/models/mlp.py`의 E03 구조 재사용. 과거 이봉헌 실험 점수는 이번 결과에 합산하지 않음 |
| 입력 | float32 `(N,128,9)`, 모델 안에서 시간 우선·채널 순서 유지하며 1,152개 값으로 펼침 |
| 채널 | body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z |
| 라벨 | 0 걷기, 1 계단 오르기, 2 계단 내려가기, 3 앉기, 4 서기, 5 눕기 |
| 모델 | Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(6, 선형 logits) |
| 파라미터 | 147,584 + 8,256 + 390 = 156,230개, 모두 학습 가능 |
| 정규화 | Dropout 0, weight decay·PCA·class weighting 사용하지 않음 |
| 초기화 | 기존 Keras Dense 기본 Glorot uniform, bias 0. 모델 생성 전에 학습 seed 설정 |
| 학습 seed | 2026, 2027, 2028. 분할은 재생성하지 않음 |
| Adam | lr 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7 |
| 손실 | sparse categorical cross-entropy, `from_logits=True` |
| 학습 | batch 64, 최대 40 epoch, 매 epoch fit 행 shuffle, 마지막 불완전 batch 유지 |
| 최선 가중치 | val_loss 최소, min_delta 0, patience 6, 최선 가중치 복원 |
| 검증 | 6클래스 Accuracy·Macro F1, zero_division 0, 혼동행렬 행=정답·열=예측 |
| 집계 | 세 seed를 모두 포함한 평균·표본 SD(ddof=1). 최고 seed 선택·추가 튜닝 없음 |
| 실행 환경 | Windows CPU, TensorFlow 2.21.0/Keras 3.15.1, intra-op 4/inter-op 1, deterministic ops |
| 시간 비교 | 개인 PC에서 다른 모델과의 속도 순위는 산출하지 않음 |

## 데이터·전처리 고정

- 데이터 SHA-256: `00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459`.
- 기존 LSTM과 동일한 원본 통계 SHA-256: `42b1b895d420a21ed7cce3b3636a4cbed96350953b27057c0670db9062d79fb2`.
- 통계는 공개 `bongheon-lstm-v1-full.zip`의 `data/preprocess.npz`에서 원본 바이트를 복원한다. ZIP SHA-256은 `f97fcdd145cbd0f6ac777ff68d576a2b7fbf832d6684e1fa36073d3b48f747a0`이다. `scripts/prepare_mlp_inputs.py`가 두 해시를 검증한다.
- 첨부된 Xs는 사람 분할·센서 구간 처리가 완료된 배열이지만 표준화 적용 전이다. 기존 fit 통계를 새로 추정하거나 PCA를 실행하지 않고, 저장된 float32 `(x-mean)/std`를 정확히 한 번 적용한다. loader는 전체 fit으로 통계를 대조한다.
- fit: 5,577개, 사람 `[3,5,7,8,11,15,16,17,19,22,25,26,27,28,29,30]`.
- validation: 1,775개, 사람 `[1,6,14,21,23]`. fit/validation 사람 및 sample_id가 겹치지 않음을 검사한다.
- sample_id는 첨부 캐시의 0 시작 `train:NNNNNN`을 유지한다. 과거 참조 로더의 1 시작 ID로 바꾸지 않는다.
- test의 shape/dtype와 사람 목록은 최초 파일 규격 점검 때 확인했다. 학습·재로딩·오류 분석 loader는 `test_*`를 읽지 않으며, test 예측·성능 계산·모델 선택은 수행하지 않는다.
- 561특징 Xf는 모델 입력에 사용하지 않는다. 전처리 인계는 정회서 담당 자료이고, 이번 작업은 해당 캐시를 MLP에 연결한 새 실행이다.

## 저장과 재현

각 실행의 `config.json`, `split.json`, `history.csv`, `model.keras`, `preprocess.npz`, `metadata.json`, `validation_predictions.csv`, `metrics.json`, `reload_check.json`, `fresh_process_verification.json`을 함께 보관한다. `metadata.environment.git_commit`은 원격에서 읽은 기반 커밋이며, 로컬 신규 코드의 실제 버전은 `code_sha256`과 고정 소스 snapshot으로 구분한다.

소량 점검은 fit 클래스별 10개·validation 클래스별 5개를 사용한 2 epoch 실행이다. 표준화 통계는 전체 fit의 기존 통계를 그대로 쓴다. 소량 점검 점수는 정식 결과에 합산하지 않는다.

저장 모델은 평가용이며, 최선 epoch 복원 가중치와 종료 시점 optimizer 상태가 정확한 이어 학습 checkpoint를 이루지는 않는다. 추론에는 `compile=False`, `training=False`를 사용하고 확률은 logits에 안정적인 softmax를 적용한다.

최종 test를 팀·지도교수 일정에 맞춰 실행할 때는 세 저장 모델과 설정을 먼저 동결하고 전부 평가해야 한다. 이번 개발 검증 결과를 독립 test 성능으로 쓰거나 test 결과를 보고 설정을 다시 고르지 않는다.
