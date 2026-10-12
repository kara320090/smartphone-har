"""Build portable explanatory notebooks; execute separately in the chosen kernel."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BOOT = '''from pathlib import Path
import os, sys, json
ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "src").is_dir():
    raise RuntimeError("Open the notebook from the repository or notebooks directory")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
'''


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Smartphone HAR", "language": "python", "name": "smartphone-har"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(notebook, ROOT / "notebooks" / name)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
save("01_backprop_validation.ipynb", [
    md("# 미분과 역전파 검증\n이봉헌 담당. 작은 손계산, NumPy 역전파, 중심차분, TensorFlow 자동미분을 비교한다. HAR 성능 실험과 별개의 수학 검증이다."),
    code(BOOT),
    md("## 한 뉴런 손계산\n입력 x=2, 가중치 w=3, 편향 b=1, 목표 t=10. 예측 wx+b=7, 손실 (예측-t)²/2=4.5. 연쇄법칙으로 dw=-6, db=-3이다. 역전파는 기울기를 구하고 optimizer는 그 기울기로 파라미터를 갱신한다."),
    code('from src.math_checks import scalar_example, run_checks\nscalar = scalar_example()\nprint(json.dumps(scalar, indent=2))'),
    md("학습률 0.1에서는 손실이 1.125로 감소하고 0.5에서는 10.125로 증가한다. 큰 학습률이 항상 빠른 수렴을 뜻하지 않는다."),
    md("## 작은 MLP의 전체 파라미터 검사\n구조는 2 → 3(tanh) → 2(logits)이며 가중치와 편향을 합쳐 17개다. 평균 교차엔트로피에서 출력 기울기는 (P−Y)/B이고, dW=XᵀD, db=sum(D)로 계산한다. 배치 크기로 두 번 나누지 않는다.\n\n중심차분은 [L(θ+h)−L(θ−h)]/(2h)이다. float64에서 h=10⁻³,10⁻⁴,10⁻⁵를 모두 검사한다. tanh를 사용해 ReLU의 0 지점 미분 문제를 피한다."),
    code('summary = run_checks(ROOT / "reports/math")\nprint(json.dumps(summary, indent=2))\nassert summary["passed"]'),
    code('import csv\nrows = list(csv.DictReader((ROOT / "reports/math/gradient_details.csv").open()))\nprint("Checks:", len(rows), "= 17 parameters x 3 step sizes")\nfor row in rows[:6]:\n    print(row)'),
    md("## 설명할 내용\n상대 오차의 분모는 max(10⁻⁸, |수치 기울기|+|해석 기울기|)이다. h를 작게 하면 절단 오차는 줄지만 너무 작으면 반올림 오차가 커진다. 일치 확인은 이 작은 예제의 미분 구현 검증이며 전체 학습의 일반화 성능을 입증하지 않는다."),
])
save("02_training_walkthrough.ipynb", [
    md("# MLP 공통 학습과 저장\n실제 UCI 데이터의 작은 부분으로 로딩 → 학습 → 저장 → 재로딩을 연결한다. 이 노트북은 연결 검사이며 정식 성능은 reports/RESULTS_KO.md에 따로 기록한다."),
    code(BOOT + '\nDATA_ROOT = Path(os.getenv("HAR_DATA_ROOT", str(ROOT / "data/raw/UCI HAR Dataset")))\nif not DATA_ROOT.exists():\n    raise FileNotFoundError("Run scripts/download_data.py or set HAR_DATA_ROOT")'),
    md("## 사람별 분할과 데이터 계약\n검증 사람 1·6·14·21·23을 고정한다. 시계열 X_seq는 (N,128,9), 가공 특징 X_features는 (N,561), y는 0~5다. 원 파일 행 번호는 1부터 세는 sample_id로 유지한다. 공식 시험 데이터는 로드하지 않는다."),
    code('from src.adapters.uci_reference import load_dataset\nfrom src.contracts import prepare_data\nfrom src.runtime import configure_runtime\nfrom src.models import build_mlp\nfrom src.train import train_model, predict_batches\nsamples, indices = load_dataset(DATA_ROOT)\nprint({k: len(v) for k,v in indices.items()})\nconfig = json.loads((ROOT / "configs/E02.json").read_text())\ndata = prepare_data(samples, indices, config, smoke=True)\nprint("Smoke shapes:", data.x_fit.shape, data.x_validation.shape)'),
    md("## 모델과 손실\n561 → 128 ReLU → 64 ReLU → 6 logits. 80,582개 파라미터. 마지막 층은 softmax를 적용하지 않으며 손실 함수에서 from_logits=True를 사용한다. 표준화는 학습 부분에서만 계산하고 검증에는 그대로 적용한다."),
    code('configure_runtime(config["seed"])\nmodel = build_mlp(data.x_fit.shape[1:])\nmodel.summary()'),
    code('from datetime import datetime\nconfig["epochs"] = 2\nconfig["verbose"] = 0\nconfig["run_directory"] = str(ROOT / "runs" / ("notebook_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))\nrun = train_model(model, data, config)\nprint(json.dumps(json.loads((run / "metrics.json").read_text()), indent=2))'),
    md("## 원 입력에서 저장물 재사용\n새 세션에서 전처리 통계와 모델을 읽고, 같은 원 입력을 변환해 동일 예측인지 확인한다. 정답은 추론 입력에 들어가지 않는다. training=False는 Dropout을 끈다."),
    code('import numpy as np\nfrom tensorflow import keras\nfrom src.adapters.preprocessing_reference import Preprocessor\npre = Preprocessor.load(run / "preprocess.npz")\nrestored = keras.models.load_model(run / "model.keras", compile=False)\nx = pre.transform(data.raw_validation)\nnp.testing.assert_allclose(predict_batches(model, data.x_validation), predict_batches(restored, x), rtol=1e-5, atol=1e-6)\nprint("PASS: raw input -> saved preprocessing -> reloaded model")'),
    md("## 공통 학습 기능\n학습에서만 shuffle을 사용한다. cache는 배열로 만든 dataset의 반복 읽기를 줄이고, batch는 한 번의 갱신 단위, prefetch는 다음 배치를 준비한다. 정식 설정은 최대 40 epoch와 val_loss patience 6이며 최저 검증 손실의 가중치를 복원한다. 검증 macro F1는 여섯 행동을 같은 비중으로 평균한다.\n\n공통 trainer는 다른 담당자의 Keras 분류 모델도 받는다. 오토인코더는 task_type='reconstruction'을 지정하고 정답 대신 입력 자체를 목표로 사용한다. 해당 모델을 구현하는 책임은 ③에게 있다."),
])
save("03_condition_comparison.ipynb", [
    md("# 학습 조건 비교와 해석\n동일 분할·seed 2026에서 E02·E03·E07–E11 결과를 비교한다. 이 노트북은 저장 결과를 읽으며 시험 데이터로 모델을 선택하지 않는다."),
    code(BOOT),
    code('import csv\nfrom IPython.display import display, Image\nrows = list(csv.DictReader((ROOT / "reports/experiment_summary.csv").open(encoding="utf-8")))\nfor row in rows:\n    print(row)'),
    code('display(Image(filename=str(ROOT / "reports/learning_curves.png")))\ndisplay(Image(filename=str(ROOT / "reports/validation_f1.png")))'),
    md("## 바꾼 것과 유지한 것\nE07은 표준화만 제거, E08은 두 은닉층에 Dropout 0.3, E09와 E10은 학습률만 변경한다. E11은 PCA 95%를 사용하며 입력 차원과 첫 층 파라미터 수가 함께 달라진다. E03은 특징 벡터 대신 시계열 입력을 사용하므로 구조만의 비교가 아니다."),
    code('baseline = float(rows[0]["macro_f1"])\nfor row in rows[1:]:\n    print(row["experiment"], "delta macro F1 (percentage points):", round(100*(float(row["macro_f1"])-baseline), 2))'),
    md("## 질문에 답하기\n1. 검증 손실이 최소인 epoch와 학습을 멈춘 epoch는 왜 다른가?\n2. 학습 손실은 감소하지만 검증 손실이 증가하면 어떤 원인을 의심하는가?\n3. Dropout 또는 작은 학습률의 효과가 이 데이터와 seed에서 어떻게 나타났는가?\n4. PCA의 큰 분산 보존이 높은 분류 성능을 보장하지 않는 이유는 무엇인가?\n5. 반복 seed와 공식 시험 평가 없이 주장할 수 있는 범위는 어디까지인가?\n\n관측값은 reports/RESULTS_KO.md와 대조한다. 단일 seed 결과를 전체 팀 최종 모델 선택이나 통계적 우월성으로 확대하지 않는다."),
])
print("Created three notebooks")
