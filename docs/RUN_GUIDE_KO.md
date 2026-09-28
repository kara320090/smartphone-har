# 설치와 실행 상세 안내

② 기준모델 구현·학습 담당의 코드와 실제 실험 결과다. 공식 UCI HAR 데이터로 특징 MLP·시계열 MLP, 공통 학습·저장, 역전파 검증, E07–E11 조건 비교를 실행한다. 팀의 데이터 담당자 작업을 기다리지 않고 실행할 수 있도록 교체 가능한 참조 데이터 어댑터를 포함했다.

먼저 [측정 결과](../reports/RESULTS_KO.md), [팀 인계 방법](HANDOFF_KO.md), [개인 발표 원고](PRESENTATION_KO.md)를 읽는다. 실행과 결과를 직접 확인하며 설명할 수 있도록 [노트북](../notebooks/)도 제공한다.

## 포함된 작업

| 담당 작업 | 파일 | 확인 방법 |
|---|---|---|
| E02 특징 MLP·E03 시계열 MLP | `src/models/mlp.py` | 파라미터 80,582개·156,230개, 6 logits |
| 공통 학습·저장·조기 종료 | `src/train.py` | 분류 모델과 재구성 모델 연결 검사 |
| 데이터 연결 어댑터 | `src/adapters/`, `src/contracts.py` | 사람 분할, train-only 통계, 표본 정렬 |
| 조건 비교 7개 | `configs/`, `src/experiments.py` | 저장 설정·곡선·검증 예측 CSV |
| 미분·역전파 검증 | `src/math_checks.py` | 손계산·NumPy·중심차분·자동미분 |
| Data API·TFRecord | `src/train.py`, `src/tfrecord_check.py` | 실제 학습 200개 표본의 완전 동일 왕복 |
| 모델·전처리 재사용 | `src/verify_bundle.py` | 새 프로세스에서 원 입력부터 확률 재현 |
| 환경·설명·인계 | `requirements*.txt`, `docs/`, `notebooks/` | 다른 팀원이 재실행할 절차 |

학습 결과는 **검증 5명에 대한 seed 2026 결과**다. 공식 시험 성능이나 팀 전체 모델의 최종 우승 결과가 아니다. 실제 사용한 장치·Python·라이브러리 버전은 각 실행의 `metadata.json`에 기록한다.

## 설치와 실행

저장소 루트에서 실행한다. 검증 환경은 Windows CPU와 Python 3.11이며 TensorFlow는 `from tensorflow import keras`로 통일했다. 다른 OS·Colab에서의 재실행은 팀에서 확인할 항목이다. Python 3.14 환경 대신 아래처럼 3.11을 명시한다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_data.py
```

데이터는 약 60MB의 공식 배포 ZIP을 받아 압축 해제한다. `Inertial Signals`를 포함하므로 시계열 MLP도 실행할 수 있다. 원 데이터는 Git에 올리지 않는다. 다운로드 시 계획서의 SHA256과 비교한다.

먼저 작은 실제 데이터로 연결을 확인한다.

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/E02.json --smoke --runs-dir runs/smoke
```

학습 60개·검증 30개, 2 epoch의 연결 검사다. 정식 점수로 사용하지 않는다. 정식 7개 실험은 아래 명령으로 실행한다.

```powershell
.\.venv\Scripts\python.exe -m src.experiments --data-root "data/raw/UCI HAR Dataset" --runs-dir runs/reproduced
.\.venv\Scripts\python.exe -m src.report --runs-dir runs/reproduced --output-dir reports/reproduced
.\.venv\Scripts\python.exe -m src.verify_bundle --data-root "data/raw/UCI HAR Dataset" --runs-dir runs/reproduced --report reports/reproduced/fresh_process_verification.json
```

이미 존재하는 실행 폴더는 덮어쓰지 않는다. 다시 실행할 때 `--runs-dir`에 새로운 이름을 사용한다. 일부 실험만 하려면 `--experiments E02 E03`, 단일 실험은 `python -m src.train --config configs/E09.json ...`을 사용한다. `src.train --seed 2027`로 ④의 반복 실행도 가능하다. 비교표 자동 생성은 이번 담당 범위인 7개의 seed 2026 실행을 대상으로 한다.

검증 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.math_checks --output-dir reports/math
.\.venv\Scripts\python.exe -m src.tfrecord_check --data-root "data/raw/UCI HAR Dataset"
```

노트북은 위 환경의 Python 커널로 연다. 기본 데이터 경로는 `data/raw/UCI HAR Dataset`이며 다른 위치는 `HAR_DATA_ROOT` 환경 변수로 지정한다. 노트북의 학습 셀은 실행마다 새 smoke 폴더를 생성한다.

`requirements.txt`는 주요 패키지를 고정했고 `requirements.lock.txt`는 실제 검증 환경의 전체 설치 목록이다. 동일 Windows 환경을 더 엄격하게 재현하려면 후자를 설치한다. 다른 OS에서 전체 lock의 호환성은 별도 확인한다.

## 데이터와 실험 조건

- 공식 train 7,352개에서 검증 사람 1·6·14·21·23을 분리한다. 학습 16명 5,577개, 검증 5명 1,775개다.
- 공식 test 9명 2,947개는 보존한다. 기본 로더는 시험 배열을 읽지 않고, 이번 학습·설정 선택에도 사용하지 않는다.
- 시계열은 `float32 (B,128,9)`, 특징은 `float32 (B,561)`, 정답은 정수 `(B,)`, 범위 0–5다.
- 채널 순서는 body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z다. sample_id는 `train:000001`처럼 공식 split과 1부터 시작하는 원 행 번호다.
- 특징은 열별, 시계열은 학습 표본·시간을 합쳐 채널별로 표준화한다. 검증 데이터에서 평균·표준편차·PCA를 다시 학습하지 않는다.
- 기본 Adam 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, batch 64, 최대 40 epoch, val_loss patience 6. 최저 검증 손실 가중치를 복원한 뒤 macro F1를 계산한다.
- 분류기는 6 logits를 출력한다. 교차엔트로피는 `from_logits=True`이며 softmax는 평가·추론 경계에서 적용한다.
- seed 2026과 결정적 연산 설정을 사용한다. 장치·버전까지 달라지는 경우 완전히 같은 수치를 보장하지는 않는다.

| 실험 | 기준 대비 변경 |
|---|---|
| E02 | 561 → 128 ReLU → 64 ReLU → 6 |
| E03 | 시계열 128×9를 모델 내부에서 펼침 → 128 → 64 → 6 |
| E07 | E02에서 표준화 제거 |
| E08 | E02의 두 은닉층 뒤 Dropout 0.3 |
| E09 | E02 학습률 0.0003 |
| E10 | E02 학습률 0.003 |
| E11 | E02 입력을 학습 표준화 후 PCA 95%로 변환 |

## 저장물

학습 모델 7개는 [GitHub 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1)의 `smartphone-har-bongheon-complete.zip`에 포함된다. ZIP을 풀어 나온 `smartphone-har` 폴더에서 실행하거나, 그 안의 `runs/formal`을 clone한 저장소의 같은 경로로 복사한다. 릴리스 ZIP에도 원 데이터와 Python 가상환경은 포함하지 않으므로 설치와 데이터 다운로드를 먼저 진행한다.

`runs/<묶음>/E02_seed2026/` 등 실행 폴더마다 다음을 남긴다.

- `model.keras`: 최저 검증 손실 시점의 모델 가중치. 평가·추론은 `compile=False`로 로드한다.
- `preprocess.npz`: 입력 종류, 표준화 통계, PCA 축·평균·설명분산. pickle 없이 읽는다.
- `metadata.json`, `config.json`: 채널·클래스 순서, 입력 shape, 코드 해시, 모델·전처리 해시, 환경·학습 조건.
- `split.json`: 실제 사용한 sample_id와 사람 목록. smoke 실행은 그 부분집합을 기록한다.
- `history.csv`, `metrics.json`, `validation_predictions.csv`: epoch 기록과 검증 결과, 각 표본의 정답·예측·logits·확률.
- `initial_gradient_check.json`, `reload_check.json`: 초기 기울기 유한값 검사와 저장 전후 일치 검사.
- `status.json`: running/complete/failed. 실패한 실행은 완료 결과에 포함하지 않는다.

Keras 파일에는 optimizer 상태도 포함된다. 가중치는 best epoch로 복원하지만 optimizer 상태는 마지막 학습 시점이므로 **정확한 중단 지점 학습 재개용 저장물은 아니다**. 재현 실험은 고정 설정과 seed로 처음부터 학습한다.

## 역할 경계

①의 분석·데이터 사전·군집 작업, ③의 CNN/RNN/LSTM/AE 구현, ④의 전체 모델 반복·최종 시험 평가, ⑤의 완성 시연·지연시간 측정은 각 담당자가 수행한다. 참조 어댑터를 통해 독립 실행을 확인했으며, 팀원 구현이 도착하면 [HANDOFF_KO.md](HANDOFF_KO.md)의 항목을 맞춰 연결한다. 팀원의 실제 환경에서 실행했다는 확인이나 개인의 이해·발표 연습을 대신 완료했다고 표시하지 않는다.

## 출처

- [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), Reyes-Ortiz 외, DOI 10.24432/C54S4K, CC BY 4.0.
- [TensorFlow 설치](https://www.tensorflow.org/install/pip), [자동미분](https://www.tensorflow.org/guide/autodiff), [tf.data](https://www.tensorflow.org/guide/data), [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord).
- [Keras 저장](https://keras.io/guides/serialization_and_saving/), [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
- 실험 분할·구조·ID·완료 기준: 제공된 스마트폰 행동 인식 10주 상세계획서 06·09·10·13–15·18·20·21·25·26·30절.
