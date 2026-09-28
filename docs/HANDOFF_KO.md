# 팀 모듈 인계와 연결

이봉헌 파트는 입력을 준비한 뒤 `train_model(model, data, config)`를 호출하면 모델 종류에 관계없이 같은 학습·저장 형식을 사용하도록 구현했다. 현재 실행은 참조 UCI 어댑터를 사용했다. 다른 팀원의 파일을 덮어쓰지 않도록 `src/data.py`, `src/preprocessing.py`, `src/models/registry.py`, `src/evaluate.py`, `src/inference.py`는 생성하지 않았다.

## ① 데이터 담당자와 맞출 내용

`load_dataset(root) -> samples, split_indices` 형태를 사용한다. `samples`는 다음 키를 가진 딕셔너리다.

| 키 | dtype·shape | 의미 |
|---|---|---|
| X_features | float32 (N,561) | 전처리 전 공식 특징 |
| X_seq | float32 (N,128,9) | 전처리 전 공식 구간 신호 |
| y | int64 (N,) | 원 라벨에서 1을 뺀 0–5 |
| subject | int64 (N,) | 사람 ID, 모델 입력에서 제외 |
| sample_id | 문자열 (N,) | train:000001 등 1부터 세는 원 행 번호 |

`split_indices`는 `fit`, `validation`, `test` 키에 해당하는 행 인덱스 배열이다. 공식 train만 읽을 때 `test`는 빈 배열이다. 검증 ID는 1·6·14·21·23이며 모든 배열에 동일한 인덱스를 적용한다.

```python
from src.contracts import prepare_data
from src.runtime import configure_runtime
from src.models import build_mlp
from src.train import train_model

# 팀 모듈이 준비되면 이 한 줄의 import를 바꾼다.
from src.adapters.uci_reference import load_dataset

samples, splits = load_dataset(data_root)
data = prepare_data(samples, splits, config)
configure_runtime(config["seed"])
model = build_mlp(data.x_fit.shape[1:], config["dropout"])
config["run_directory"] = "runs/team/E02_seed2026"
run_directory = train_model(model, data, config)
```

`prepare_data`는 **원 입력**을 받는다. 이미 표준화한 입력에 다시 적용하면 이중 표준화가 되므로 사용하지 않는다. ①이 전처리까지 제공할 경우 `TrainingData`를 직접 구성하고, `preprocessor.save(path)`, `transform(raw)`, `input_kind` 계약을 맞춘다. 현재 `verify_bundle.py`는 `reference_npz_v1` 형식을 읽으므로 팀 저장 형식이 다르면 로더와 metadata의 preprocessing 필드를 함께 수정해야 한다.

참조 NPZ 형식은 `schema_version=1`, `input_kind`, `standardize`, `mean`, `scale`, `components`, `pca_mean`, `explained_variance_ratio`다. 순서는 `(x-mean)/scale`, PCA 사용 시 `(z-pca_mean) @ components.T`다. PCA를 사용하지 않으면 components의 행 수는 0이다. mean/std는 float64로 추정하며 변환 결과는 float32다. 0 표준편차는 1로 처리한다. 시계열 통계 축은 `(N,T)`다.

팀 연결 시 원 입력·분할·통계를 기존 결과와 비교한다. 서로 다른 정밀도·PCA 구현·버전을 사용해 숫자가 달라지면 새 실행 ID로 재학습한다. 이전 모델에 새로운 전처리를 붙이지 않는다.

## ③ 비교 모델 담당자와 맞출 내용

`train_model`은 MLP 전용 함수가 아니다. Keras 모델 객체를 직접 받는다. CNN/RNN/LSTM에는 `TrainingData.x_fit`이 (N,128,9), 모델 입력이 (None,128,9), 출력이 (None,6)인 데이터를 전달한다. seed를 **모델 생성 전에** 설정한다. 모델 마지막 층은 softmax 없는 logits다.

오토인코더는 `task_type="reconstruction"`을 지정한다. 입력과 같은 shape의 복원값을 출력하고 MSE·재구성 지표를 저장한다. y는 데이터 계약의 라벨 확인용이며 재구성 손실의 목표로 쓰지 않는다. 현재 CLI는 봉헌 담당 MLP 실험을 생성하므로 다른 모델의 registry 연결은 ③이 함수 호출 부분에 추가한다.

공통 trainer의 외부 모델 연결은 작은 선형 분류기와 작은 재구성 모델로 검증했다. ③의 실제 CNN/RNN/LSTM/AE 모델 자체는 아직 구현·검증한 것으로 간주하지 않는다.

## ④ 평가 담당자에게 넘길 내용

- 7개 실험의 실제 config, history, validation_predictions, metrics와 모델·전처리 저장물.
- `validation_predictions.csv`의 클래스 인덱스는 0–5이고 확률 열은 probability_0부터 probability_5까지다.
- 지표는 6개 클래스를 항상 포함한 macro F1, accuracy, class recall, 혼동행렬이다. 혼동행렬 행은 정답·열은 예측이며 zero_division=0이다.
- 최선 epoch는 val_loss로 선택한다. 설정 선택에는 복원 모델의 검증 macro F1를 사용한다.
- seed 2027·2028 반복은 동일 함수를 새 run_directory와 seed로 호출한다. 기본 6개 모델 및 별도 선택 설정의 반복·평가 책임은 ④에게 있다.
- 이번 결과에는 공식 시험 점수가 없다. 팀 설정 확정 후 공식 test를 평가하고, 결과를 보고 설정을 다시 선택하지 않는다.

## ⑤ 추론 담당자에게 넘길 내용

`model.keras`, `preprocess.npz`, `metadata.json`, `config.json`, `split.json`을 **같은 실행 폴더 단위**로 전달한다. `keras.models.load_model(path, compile=False)` 후 `model(x, training=False)`를 사용한다. softmax는 float64로 안정적으로 계산할 수 있다.

561 특징 모델에는 공식 561 특징 벡터가 필요하다. 새로운 파형 하나에서 이 특징을 자동 계산하는 기능은 포함하지 않았다. sample_id로 파형과 대응 특징을 함께 찾아야 한다.

`src.verify_bundle`은 원 검증 입력에서 저장 전처리를 적용하고 모든 검증 표본의 확률·라벨·지표를 재현한다. 이 검사는 학습 저장물 인계 확인이다. ⑤는 이를 받아 화면, 잘못된 입력 처리, 실행 시간 측정, 규칙·관계표 기능을 개발한다.

## 팀장 확인표

- [ ] ①의 공식 로더와 참조 로더가 같은 sample_id·배열·분할을 반환하는지 공동 확인
- [ ] ③의 모델 생성 함수가 같은 입력·logits 계약을 따르는지 연결
- [ ] ④가 공통 설정·지표로 전체 모델 비교와 반복 실행
- [ ] ⑤가 저장 폴더를 받아 자기 환경에서 예측 재현
- [ ] 팀 장치 예약·실험 실행 ID·오류·수정 내역 공유
- [ ] 각자 자기 파트의 근거와 한계를 작성하고 서로 설명 확인

위 항목은 실제 팀 협업이 필요하므로 로컬 구현만으로 완료 표시하지 않는다.
