> 업로드 준비 시 확인: 현재 main의 CNN은 85,574개 파라미터이며 이 보고서의 비교용 C01(14,054개)과 다릅니다. 본 문서의 네 모델 표는 유재윤 작업에서 수행한 보조 비교입니다. main의 담당자 공식 결과표를 대체하지 않습니다. 당시 실행 코드·모델은 RNN Release, 숫자 증빙은 reports/rnn/run_evidence/에 보존합니다.

# 유재윤 스마트폰 행동 인식 RNN 수행 보고서

작성일 2026년 10월 12일 · 담당 RNN 구현·학습·평가 및 네 모델 성능 비교

RNN 기준 R01의 validation Macro F1은 **0.7191 ± 0.0586**, Accuracy 평균은 **73.56%**다. 세 조건을 각각 세 seed에서 실행한 개발 튜닝에서는 **R02**을 선택했다. 아래 수치는 저장 예측에서 재계산한 검증 결과이며 공식 test 성능을 뜻하지 않는다.

## 연구 질문과 실제 수행 범위

동일한 스마트폰 센서 구간을 기본 순환신경망으로 처리할 때 여섯 활동을 어느 정도 구분하는지, 학습 seed·은닉 크기·규제에 따라 결과가 어떻게 달라지는지 확인했다. RNN 모델, 설정, 기존 공통 trainer 연결, 반복 실행, 재로딩, 오류·사람별 분석을 이번 작업에서 수행했다. 네 모델의 비교 자료를 마련하기 위해 기존 main MLP·LSTM과 계획서 CNN을 같은 제공 데이터로 추가 실행했다. 이 보조 실행을 다른 담당자의 수행 실적으로 표기하지 않는다.

## 입력과 사람 분할

정회서의 전처리 코드와 제공된 `processed_data.npz`를 사용했다. 이 캐시의 시계열은 표준화되지 않은 float32 `(N,128,9)`다. body_acc xyz, body_gyro xyz, total_acc xyz 순서이며 시간축은 128, 채널축은 9다. fit 16명 5,577개와 validation 피험자 1·6·14·21·23의 1,775개가 분리되어 있다. 사람·sample_id가 서로 겹치지 않으며 fit·validation에서 비유한 값은 발견되지 않았다. 저장된 통계가 full fit에서 계산한 값과 정확히 일치함을 확인하고 한 번만 적용했다. PCA·561특징은 RNN 입력에 사용하지 않았다. sample_id는 원본 공식 train의 0 시작 행 번호를 유지했다.

## 순환 상태와 구조

`keras.layers.SimpleRNN`은 `h_t=tanh(x_t W_x+h_(t-1) W_h+b)`로 이전 상태와 현재 입력을 결합한다. 구간마다 `h_0=0`, `stateful=False`로 처리하고 마지막 `h_128`에서 Dense 6의 logits를 계산한다. 다른 구간이나 이전 호출의 상태를 이어 사용하지 않는다. R01은 은닉 32개이며 `9×32+32×32+32+32×6+6=1,542`개 파라미터다. tanh, Glorot 입력 가중치, 직교 순환 가중치, 0 편향을 사용한다. 입력·순환 Dropout은 0이며 R03에서만 마지막 출력에 Dropout 0.3을 적용했다.

## 학습과 선택 조건

Adam 학습률 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7, batch 64, 최대 40 epoch, val_loss patience 6과 최저 손실 가중치 복원을 사용했다. 손실은 logits 기반 sparse cross entropy이며 fit에서만 shuffle한다. seed 2026·2027·2028을 각 모델 구성 전에 설정하고 결정적 연산을 켰다. clipping은 사용하지 않았다. R03은 같은 seed의 R02 학습 전 가중치를 복사했다. 세 seed의 validation Macro F1 평균으로 RNN 조건을 선택하고, 동률이면 적은 파라미터·실험 ID 순서로 정했다. 시연 seed는 2026으로 유지한다. 실제 환경은 Python 3.12.14, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.2.6, Windows CPU다.

## RNN 결과와 선택 이유

| 조건 | 모델 | 파라미터 | Accuracy 평균 ± SD | Macro F1 평균 ± SD |
|---|---|---:|---:|---:|
| R01 | rnn | 1,542 | 0.7356 ± 0.0497 | 0.7191 ± 0.0586 |
| R02 | rnn | 5,126 | 0.7855 ± 0.0605 | 0.7715 ± 0.0675 |
| R03 | rnn | 5,126 | 0.7654 ± 0.0670 | 0.7525 ± 0.0804 |

선택 조건 R02의 validation Macro F1 평균은 0.7715다. seed SD는 ddof=1 표본 표준편차이며 고정된 한 사람 분할에서 초기화·섞기의 변동을 나타낸다. 일반화 신뢰구간으로 사용하지 않는다. 검증 손실로 checkpoint를 선택하고 같은 validation에서 구성을 선택했으므로 이 점수는 개발 결과다. RNN 튜닝 승자와 기준 R01을 구분해 보관했다.

| run_id | Accuracy | Macro F1 | 최선 epoch | 실행 epoch |
|---|---:|---:|---:|---:|
| R01_seed2026 | 0.6879 | 0.6621 | 7 | 13 |
| R01_seed2027 | 0.7318 | 0.7160 | 17 | 23 |
| R01_seed2028 | 0.7870 | 0.7792 | 22 | 28 |
| R02_seed2026 | 0.7301 | 0.7063 | 6 | 12 |
| R02_seed2027 | 0.7763 | 0.7670 | 10 | 16 |
| R02_seed2028 | 0.8501 | 0.8412 | 11 | 17 |
| R03_seed2026 | 0.6901 | 0.6626 | 4 | 10 |
| R03_seed2027 | 0.8186 | 0.8175 | 23 | 29 |
| R03_seed2028 | 0.7876 | 0.7773 | 11 | 17 |

![선택 조건 seed 2026 학습 곡선](../../reports/rnn/figures/R02_seed2026_history.png)

가로축은 epoch, 왼쪽 세로축은 교차엔트로피, 오른쪽은 Accuracy·Macro F1이다. 점선은 최저 validation loss checkpoint다. 곡선 원자료는 해당 실행의 history.csv에 있다.

## 네 모델의 같은 입력 비교

| 조건 | 모델 | 파라미터 | Accuracy 평균 ± SD | Macro F1 평균 ± SD |
|---|---|---:|---:|---:|
| M01 | mlp | 156,230 | 0.8667 ± 0.0117 | 0.8646 ± 0.0125 |
| C01 | cnn | 14,054 | 0.8792 ± 0.0068 | 0.8784 ± 0.0073 |
| R01 | rnn | 1,542 | 0.7356 ± 0.0497 | 0.7191 ± 0.0586 |
| L01 | lstm | 21,222 | 0.8832 ± 0.0194 | 0.8813 ± 0.0195 |

모든 행은 이번 작업에서 같은 제공 NPZ·전처리·사람 분할·평가 코드로 새로 실행한 세 seed 결과다. M01·C01·L01은 RNN 담당의 비교용 보조 재현 실험이다. 기존 main에 기록된 과거 점수를 그대로 가져오지 않았다. MLP는 128×9를 내부에서 펼쳐 128·64 은닉층을 사용한다. CNN은 계획서 그림의 Conv32 k5 → Pool2 → Conv64 k5 → GAP → Dense32 ReLU → Dropout 0.2 → Dense6 구조다. LSTM은 main의 은닉 64 → Dense32 ReLU → Dense6이며 계획서의 단순 은닉 32 LSTM과 다르다. 각 모델의 파라미터·연산량·탐색 예산이 달라 차이를 하나의 구조 원인이나 각 구조의 최적 성능으로 단정할 수 없다. 동일 seed별 차이는 paired_differences.csv에 저장했다.

## 클래스와 사람별 오류 분석

선택 조건 R02_seed2026에서 가장 큰 방향성 오분류는 정답 **걷기**를 **계단 오르기**로 예측한 **143개**다. 이 사실은 혼동행렬에서 관찰한 결과이며 센서 위치·특정 주파수·기울기 문제를 원인으로 확정한 것은 아니다. 전체 오류 sample_id·사람·확률을 errors.csv에 남겼고 클래스별 precision·recall·F1·support와 사람별 성능을 별도 CSV로 저장했다.

![선택 조건 혼동행렬](../../reports/rnn/figures/R02_seed2026_confusion.png)

행은 정답, 열은 예측이며 셀은 구간 수다. 행 기준 비율은 대응 confusion.json에 있다.

| 행동 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 걷기 | 0.7333 | 0.4783 | 0.5789 | 322 |
| 계단 오르기 | 0.4712 | 0.9258 | 0.6245 | 256 |
| 계단 내려가기 | 0.6991 | 0.3278 | 0.4463 | 241 |
| 앉기 | 0.9340 | 0.6408 | 0.7601 | 309 |
| 서기 | 0.7386 | 0.9419 | 0.8280 | 327 |
| 눕기 | 1.0000 | 1.0000 | 1.0000 | 320 |

| 검증 사람 | 구간 수 | Accuracy | 6클래스 Macro F1 |
|---|---:|---:|---:|
| 1 | 347 | 0.7637 | 0.7736 |
| 6 | 325 | 0.8400 | 0.8347 |
| 14 | 323 | 0.6130 | 0.5580 |
| 21 | 408 | 0.6446 | 0.5488 |
| 23 | 372 | 0.7984 | 0.7709 |

## 계산과 정상 동작 검증

작은 float64 tanh RNN에서 30개 파라미터를 중심차분과 자동미분으로 비교했고 최대 절대 오차는 1.875e-11였다. 실제 모델에서는 초기 손실·기울기의 유한성, 독립 구간 상태, 잘못된 shape·dtype·NaN·Inf 거부, 전체 validation 예측 재로딩을 확인했다. 초기 다섯 갱신 전 기울기 진단은 고정 fit 첫 배치를 training=False로 별도 평가한 값이다. 실제 shuffle 배치의 기울기 또는 시점별 감쇠 측정과 구분한다. 이 근거만으로 기울기 소실·폭주나 긴 기억 학습을 주장하지 않는다. 각 실행의 fresh_process_verification.json은 validation 1,775개 전체의 확률·라벨·지표 재계산을 확인한다.

제공 main은 MLP·LSTM만 모델 registry에 연결되어 있었다. RNN·CNN 생성 함수를 추가하고 공통 trainer·processed cache adapter·fresh-process verifier를 재사용했다. 데이터 배열에는 표준화가 적용되지 않았다는 저장 코드의 실제 동작을 확인해 중복 표준화를 피했다. 원본 다운로드 폴더는 보존하고 수정 사본에서 작업했다.

## 해석과 남은 작업

결과는 허리 착용 스마트폰의 여섯 활동과 고정 S0 사람 분할에서의 개발 비교다. 인접 구간이 겹치므로 각 구간을 독립 피험자 표본으로 간주하지 않는다. 다른 착용 위치·기종·활동으로의 일반화는 확인하지 않았다. 공식 test는 팀에서 대상과 시점을 확정한 뒤 별도로 평가해야 한다. 팀원의 공식 모델 인계 후에는 코드·분할·전처리·평가 조건을 다시 맞춰 비교표에 수행자를 명시한다.

코드·실험·분석·원고 생성에 Codex를 사용했다. 유재윤 담당 제출 자료로 준비한 작업 묶음이며 유재윤의 이해 확인과 최종 원고 검토가 필요하다. 원자료·모델·재현 명령을 함께 제공해 검토할 수 있게 했다. 팀원에게 받은 자료는 정회서 전처리 파일·코드와 기존 main의 MLP·LSTM·공통 실행 코드다.

## 실행과 근거

프로젝트 루트에서 `python scripts/run_rnn_suite.py`로 전체 반복과 복원을 수행한다. `python -m src.rnn_analysis`는 저장 예측에서 지표와 그림을 다시 만들며, `python scripts/build_rnn_deliverables.py`는 이 원고와 노트북을 생성한다. 상세 설치는 RNN_START_HERE_KO.md를 따른다. 코드·환경 해시와 데이터 해시는 각 실행 metadata.json에 있다.

## 참고문헌

1. Reyes-Ortiz 외, Human Activity Recognition Using Smartphones, UCI Machine Learning Repository, 2013, DOI 10.24432/C54S4K. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones — 데이터 출처와 128시점 구간 설명.
2. Keras 공식 문서, SimpleRNN layer. https://keras.io/api/layers/recurrent_layers/simple_rnn/ — 기본 RNN의 옵션과 상태 정책.
3. Keras 공식 문서, EarlyStopping. https://keras.io/api/callbacks/early_stopping/ — 검증 손실 기준 종료와 최선 가중치 복원.
4. 제공 main 코드 `src/train.py`, `src/adapters/processed_npz.py`, `src/models/lstm.py` — 실제 학습·표준화·LSTM 구현.
