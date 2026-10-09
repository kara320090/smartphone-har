# LSTM 실행과 팀 인계

현재 담당은 LSTM 이봉헌이다. 아래 명령은 저장소 루트에서 실행한다. Python은 requirements.txt의 학습 환경을 사용한다. 입력 캐시는 외부 경로로 전달하며 대용량 원 데이터와 가상환경을 Git에 추가하지 않는다.

## 소량 점검

```powershell
python scripts/run_lstm_suite.py --processed-data "C:/Users/soma/Desktop/processed_data.npz" --preprocess-stats data/preprocess.npz --runs-dir runs/lstm_smoke_new --smoke --seeds 2026
```

## 정식 9회

```powershell
python scripts/run_lstm_suite.py --processed-data "C:/Users/soma/Desktop/processed_data.npz" --preprocess-stats data/preprocess.npz --runs-dir runs/lstm_new
```

L01·L02·L03을 seed별로 순차 실행하며 L03의 학습 전 가중치 원본을 자동 지정한다. 완료된 실행은 새 프로세스에서 검증한다. 미완료 폴더가 있으면 덮어쓰지 않고 멈추므로 원인을 기록하고 새 출력 폴더를 사용한다.

## 지표와 시간 분석

```powershell
python -m src.lstm_analysis --processed-data "C:/Users/soma/Desktop/processed_data.npz" --preprocess-stats data/preprocess.npz --runs-dir runs/lstm_new --output-dir reports/lstm_new
```

9개 실행이 모두 완료되고 fresh_process_verification.json이 있어야 한다. 모델 시간은 compiled tf.function, batch 1, 준비된 Tensor, 출력 .numpy() 반환까지다. 50회 워밍업과 500회 측정이며 전처리는 별도다. 별도 작업과 병행하지 않고 측정한다.

## 저장 모델 예측

```python
import numpy as np
from src.lstm_analysis import predict_raw_bundle

with np.load("processed_data.npz", allow_pickle=False) as cache:
    raw = cache["val_Xs"][:5]  # 표준화 적용 전 구간
probabilities, classes = predict_raw_bundle("runs/lstm/L01_seed2026", raw)
```

입력 shape는 (N,128,9) float32, 채널은 body_acc xyz, body_gyro xyz, total_acc xyz다. 이 함수는 저장된 전처리를 적용하므로 이미 표준화한 배열을 주지 않는다. 실제 채널 의미를 배열 shape만으로 자동 판별할 수는 없다.

## 수학과 코드 검사

```powershell
python -m pytest -q
python -m src.lstm_math_checks --output-dir reports/lstm_math_new
```

## 팀원이 받을 자료

공동 비교에는 L01 seed 2026·2027·2028의 실행 폴더와 데이터·통계 해시를 사용한다. 각 폴더에 model.keras, preprocess.npz, config.json, metadata.json, split.json, history.csv, initial_weights.npz, initialization.json, validation_predictions.csv, metrics.json, reload_check.json, fresh_process_verification.json이 있다.

L02·L03은 보조 분석이다. 다른 모델과 비교할 때 입력 표현·사람 분할·튜닝 예산·checkpoint 선택·seed를 대조한다. 점수만 보고 추가 튜닝한 결과로 기준을 교체하지 않는다. 공식 test 성능은 이번 자료에 없다.

## CLI 추론과 원본 행 대조

```powershell
python -m analysis.lstm_predict --run-directory runs/lstm/L01_seed2026 --processed-data "C:/Users/soma/Desktop/processed_data.npz" --count 5 --output reports/prediction_demo_new.json
python -m analysis.lstm_data_audit --processed-data "C:/Users/soma/Desktop/processed_data.npz" --data-root "data/UCI HAR Dataset" --output reports/dataset_audit_new.json
python -m analysis.lstm_error_features --processed-data "C:/Users/soma/Desktop/processed_data.npz" --predictions runs/lstm/L01_seed2026/validation_predictions.csv --output-dir reports/error_features_new
```

추론 CLI는 개발 캐시의 val_Xs만 사용하는 예시다. 자신의 구간을 사용할 때는 --processed-data 대신 --input-npy windows.npy를 지정한다. 입력은 표준화 전 float32 (N,128,9)이며 실제 채널 순서와 센서 단위는 사용자가 맞춰야 한다. 저장 함수가 표준화를 적용하므로 표준화된 입력을 다시 주지 않는다. 여기서 새 스마트폰 센서 자료에 대한 정확도를 검증한 것은 아니다.
