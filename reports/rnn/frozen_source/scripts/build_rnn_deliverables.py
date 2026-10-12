"""Create a Korean contribution, evidence index, defense notes and notebook."""
import csv
import json
from pathlib import Path
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    report=ROOT/'reports/rnn'
    a=json.loads((report/'analysis.json').read_text(encoding='utf-8'))
    out=ROOT/'docs/rnn-report'
    out.mkdir(parents=True,exist_ok=True)
    summary={r['experiment']:r for r in a['summary']}
    run={r['run_id']:r for r in a['runs']}
    chosen=a['selected_rnn']
    baseline=summary['R01']
    chosen_summary=summary[chosen]
    selected_run=f'{chosen}_seed2026'
    cm=np.array(json.loads((report/f'{selected_run}_confusion.json').read_text())['counts'])
    off=cm.copy(); np.fill_diagonal(off,0)
    i,j=np.unravel_index(off.argmax(),off.shape)
    names=['걷기','계단 오르기','계단 내려가기','앉기','서기','눕기']
    env=json.loads((ROOT/f'runs/rnn/{selected_run}/metadata.json').read_text())['environment']
    math=json.loads((report/'math_verification.json').read_text())
    def table(ids):
        text='| 조건 | 모델 | 파라미터 | Accuracy 평균 ± SD | Macro F1 평균 ± SD |\n|---|---|---:|---:|---:|\n'
        for exp in ids:
            r=summary[exp]
            text+=f"| {exp} | {r['model']} | {r['parameters']:,} | {r['accuracy_mean']:.4f} ± {r['accuracy_sd']:.4f} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} |\n"
        return text
    text=f'''# 유재윤 스마트폰 행동 인식 RNN 수행 보고서

작성일 2026년 10월 12일 · 담당 RNN 구현·학습·평가 및 네 모델 성능 비교

RNN 기준 R01의 validation Macro F1은 **{baseline['macro_f1_mean']:.4f} ± {baseline['macro_f1_sd']:.4f}**, Accuracy 평균은 **{baseline['accuracy_mean']*100:.2f}%**다. 세 조건을 각각 세 seed에서 실행한 개발 튜닝에서는 **{chosen}**을 선택했다. 아래 수치는 저장 예측에서 재계산한 검증 결과이며 공식 test 성능을 뜻하지 않는다.

## 연구 질문과 실제 수행 범위

동일한 스마트폰 센서 구간을 기본 순환신경망으로 처리할 때 여섯 활동을 어느 정도 구분하는지, 학습 seed·은닉 크기·규제에 따라 결과가 어떻게 달라지는지 확인했다. RNN 모델, 설정, 기존 공통 trainer 연결, 반복 실행, 재로딩, 오류·사람별 분석을 이번 작업에서 수행했다. 네 모델의 비교 자료를 마련하기 위해 기존 main MLP·LSTM과 계획서 CNN을 같은 제공 데이터로 추가 실행했다. 이 보조 실행을 다른 담당자의 수행 실적으로 표기하지 않는다.

## 입력과 사람 분할

정회서의 전처리 코드와 제공된 `processed_data.npz`를 사용했다. 이 캐시의 시계열은 표준화되지 않은 float32 `(N,128,9)`다. body_acc xyz, body_gyro xyz, total_acc xyz 순서이며 시간축은 128, 채널축은 9다. fit 16명 5,577개와 validation 피험자 1·6·14·21·23의 1,775개가 분리되어 있다. 사람·sample_id가 서로 겹치지 않으며 fit·validation에서 비유한 값은 발견되지 않았다. 저장된 통계가 full fit에서 계산한 값과 정확히 일치함을 확인하고 한 번만 적용했다. PCA·561특징은 RNN 입력에 사용하지 않았다. sample_id는 원본 공식 train의 0 시작 행 번호를 유지했다.

## 순환 상태와 구조

`keras.layers.SimpleRNN`은 `h_t=tanh(x_t W_x+h_(t-1) W_h+b)`로 이전 상태와 현재 입력을 결합한다. 구간마다 `h_0=0`, `stateful=False`로 처리하고 마지막 `h_128`에서 Dense 6의 logits를 계산한다. 다른 구간이나 이전 호출의 상태를 이어 사용하지 않는다. R01은 은닉 32개이며 `9×32+32×32+32+32×6+6=1,542`개 파라미터다. tanh, Glorot 입력 가중치, 직교 순환 가중치, 0 편향을 사용한다. 입력·순환 Dropout은 0이며 R03에서만 마지막 출력에 Dropout 0.3을 적용했다.

## 학습과 선택 조건

Adam 학습률 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7, batch 64, 최대 40 epoch, val_loss patience 6과 최저 손실 가중치 복원을 사용했다. 손실은 logits 기반 sparse cross entropy이며 fit에서만 shuffle한다. seed 2026·2027·2028을 각 모델 구성 전에 설정하고 결정적 연산을 켰다. clipping은 사용하지 않았다. R03은 같은 seed의 R02 학습 전 가중치를 복사했다. 세 seed의 validation Macro F1 평균으로 RNN 조건을 선택하고, 동률이면 적은 파라미터·실험 ID 순서로 정했다. 시연 seed는 2026으로 유지한다. 실제 환경은 Python {env['python']}, TensorFlow {env['tensorflow']}, Keras {env['keras']}, NumPy {env['numpy']}, Windows CPU다.

## RNN 결과와 선택 이유

{table(['R01','R02','R03'])}
선택 조건 {chosen}의 validation Macro F1 평균은 {chosen_summary['macro_f1_mean']:.4f}다. seed SD는 ddof=1 표본 표준편차이며 고정된 한 사람 분할에서 초기화·섞기의 변동을 나타낸다. 일반화 신뢰구간으로 사용하지 않는다. 검증 손실로 checkpoint를 선택하고 같은 validation에서 구성을 선택했으므로 이 점수는 개발 결과다. RNN 튜닝 승자와 기준 R01을 구분해 보관했다.

| run_id | Accuracy | Macro F1 | 최선 epoch | 실행 epoch |
|---|---:|---:|---:|---:|
'''
    for r in a['runs']:
        if r['model']=='rnn':
            text+=f"| {r['run_id']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['best_epoch']} | {r['epochs_ran']} |\n"
    text+=f'''
![선택 조건 seed 2026 학습 곡선](../../reports/rnn/figures/{selected_run}_history.png)

가로축은 epoch, 왼쪽 세로축은 교차엔트로피, 오른쪽은 Accuracy·Macro F1이다. 점선은 최저 validation loss checkpoint다. 곡선 원자료는 해당 실행의 history.csv에 있다.

## 네 모델의 같은 입력 비교

{table(['M01','C01','R01','L01'])}
모든 행은 이번 작업에서 같은 제공 NPZ·전처리·사람 분할·평가 코드로 새로 실행한 세 seed 결과다. M01·C01·L01은 RNN 담당의 비교용 보조 재현 실험이다. 기존 main에 기록된 과거 점수를 그대로 가져오지 않았다. MLP는 128×9를 내부에서 펼쳐 128·64 은닉층을 사용한다. CNN은 계획서 그림의 Conv32 k5 → Pool2 → Conv64 k5 → GAP → Dense32 ReLU → Dropout 0.2 → Dense6 구조다. LSTM은 main의 은닉 64 → Dense32 ReLU → Dense6이며 계획서의 단순 은닉 32 LSTM과 다르다. 각 모델의 파라미터·연산량·탐색 예산이 달라 차이를 하나의 구조 원인이나 각 구조의 최적 성능으로 단정할 수 없다. 동일 seed별 차이는 paired_differences.csv에 저장했다.

## 클래스와 사람별 오류 분석

선택 조건 {selected_run}에서 가장 큰 방향성 오분류는 정답 **{names[i]}**를 **{names[j]}**로 예측한 **{int(off[i,j])}개**다. 이 사실은 혼동행렬에서 관찰한 결과이며 센서 위치·특정 주파수·기울기 문제를 원인으로 확정한 것은 아니다. 전체 오류 sample_id·사람·확률을 errors.csv에 남겼고 클래스별 precision·recall·F1·support와 사람별 성능을 별도 CSV로 저장했다.

![선택 조건 혼동행렬](../../reports/rnn/figures/{selected_run}_confusion.png)

행은 정답, 열은 예측이며 셀은 구간 수다. 행 기준 비율은 대응 confusion.json에 있다.

| 행동 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
'''
    for k,r in enumerate([r for r in rows(report/'class_metrics.csv') if r['run_id']==selected_run]):
        text+=f"| {names[k]} | {float(r['precision']):.4f} | {float(r['recall']):.4f} | {float(r['f1']):.4f} | {r['support']} |\n"
    text+='\n| 검증 사람 | 구간 수 | Accuracy | 6클래스 Macro F1 |\n|---|---:|---:|---:|\n'
    for r in rows(report/'subject_metrics.csv'):
        if r['run_id']==selected_run:
            text+=f"| {r['subject']} | {r['support']} | {float(r['accuracy']):.4f} | {float(r['macro_f1']):.4f} |\n"
    text+=f'''
## 계산과 정상 동작 검증

작은 float64 tanh RNN에서 {math['checked_parameters']}개 파라미터를 중심차분과 자동미분으로 비교했고 최대 절대 오차는 {math['max_absolute_error']:.3e}였다. 실제 모델에서는 초기 손실·기울기의 유한성, 독립 구간 상태, 잘못된 shape·dtype·NaN·Inf 거부, 전체 validation 예측 재로딩을 확인했다. 초기 다섯 갱신 전 기울기 진단은 고정 fit 첫 배치를 training=False로 별도 평가한 값이다. 실제 shuffle 배치의 기울기 또는 시점별 감쇠 측정과 구분한다. 이 근거만으로 기울기 소실·폭주나 긴 기억 학습을 주장하지 않는다. 각 실행의 fresh_process_verification.json은 validation 1,775개 전체의 확률·라벨·지표 재계산을 확인한다.

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
'''
    (out/'REPORT_KO.md').write_text(text,encoding='utf-8')
    compact=f'''# 유재윤 RNN 개인 원고

작성일 2026년 10월 12일 · RNN 구현 학습 평가 · 팀 내부 취합용

본 연구에서 기본 순환신경망은 동일한 128×9 센서 구간의 순차 처리 기준으로 사용하였다. Keras SimpleRNN의 은닉 상태는 현재 입력과 직전 상태를 tanh로 결합하며 각 구간 시작에서 0으로 초기화하였다. 마지막 시점의 은닉 상태를 Dense 6으로 변환해 여섯 행동의 logits를 출력하였다. 기준 R01은 은닉 32개와 총 1,542개 파라미터를 갖는다.

제공된 팀 캐시의 시계열은 표준화 전 배열이므로 학습 16명 5,577개에서 계산된 저장 통계를 한 번 적용하였다. 검증은 피험자 1·6·14·21·23의 1,775개 구간으로 고정하였다. 사람 ID는 분할과 오류 분석에만 사용하였다. 라벨은 0–5이며 원본 sample_id를 유지하였다.

Adam 학습률 0.001, batch 64, 최대 40 epoch, validation loss 기준 patience 6과 최선 가중치 복원을 적용하였다. seed 2026·2027·2028을 각각 모델 생성 전에 설정하였다. 출력은 logits이며 교차엔트로피 손실을 사용하였다. 구간 사이 상태를 이어 쓰거나 clipping을 적용하지 않았다.

기준 R01의 validation Macro F1은 {baseline['macro_f1_mean']:.4f} ± {baseline['macro_f1_sd']:.4f}, Accuracy 평균은 {baseline['accuracy_mean']:.4f}였다. SD는 고정 분할에서 세 학습 seed의 표본 SD다. 은닉 64 R02와 출력 Dropout 0.3 R03을 별도 튜닝 조건으로 실행했고 세 seed 평균 기준으로 {chosen}을 선택하였다. R03은 R02의 같은 seed 학습 전 가중치를 복사하여 비교하였다. 튜닝 결과를 기준 R01의 공동 비교 결과와 구분하였다.

같은 제공 데이터와 평가 코드로 MLP·계획서 CNN·main LSTM을 보조 재실행하였다. 네 모델의 파라미터 규모와 출력 집계가 달라 성능 차이를 순환 구조 하나의 효과로 단정하지 않았다. 이 재실행은 RNN 담당 작업의 비교 자료이며 다른 담당자의 수행 실적으로 기재하지 않는다.

작은 RNN의 중심차분 {math['checked_parameters']}개를 자동미분과 대조했고 최대 절대 오차 {math['max_absolute_error']:.3e}를 확인하였다. 잘못된 입력 거부, 구간 상태 초기화, 저장 모델의 validation 전체 재로딩과 지표 재계산도 확인하였다. 선택 조건 seed 2026의 가장 큰 오분류는 {names[i]}에서 {names[j]}로의 {int(off[i,j])}개였다. 초기 고정 배치의 기울기 norm은 수치 검증이며 장기 기억 또는 시점별 소실을 증명하지 않는다.

이번 결과는 개발 validation 결과이며 공식 test는 미평가다. 최종 평가 시점·대상은 팀 합의 후 고정해야 한다. 데이터·공통 코드는 팀에서 제공받았고 RNN 코드·실험·분석·원고는 Codex 보조로 생성하였다. 최종 제출 전에 담당자의 코드·수치·설명 검토가 필요하다. 근거는 docs/rnn-report/REPORT_KO.md, reports/rnn/, runs/rnn/에 보관하였다.
'''
    (out/'contribution.md').write_text(compact,encoding='utf-8')
    (out/'evidence_index.md').write_text('''# RNN 주장과 근거

| 주장 | 근거 |
|---|---|
| 제공 NPZ·저장 통계 일치와 비누수 | reports/rnn/dataset_audit.json |
| 1,542개 기본 RNN과 상태 초기화 | src/models/rnn.py · tests/test_rnn.py |
| 사전 후보와 선택 규칙 | docs/RNN_TRAINING_PROTOCOL_KO.md · configs/R01–R03.json |
| seed별 지표 재계산 | reports/rnn/runs.csv · runs/rnn/*/validation_predictions.csv |
| 클래스·사람별 성능과 오류 | reports/rnn/class_metrics.csv · subject_metrics.csv · errors.csv |
| 계산 정확성 | reports/rnn/math_verification.json |
| 전체 validation 재로딩 | runs/rnn/*/fresh_process_verification.json |
| 초기 고정 배치 진단 | runs/rnn/R*/early_gradient_probes.json |
| 네 모델 동일 입력 보조 비교 | reports/rnn/four_model_comparison.csv · paired_differences.csv |
| 실행 코드 테스트 | reports/rnn/tests.xml |
''',encoding='utf-8')
    (out/'DEFENSE_QA_KO.md').write_text(f'''# RNN 발표 질의응답

1. RNN과 LSTM의 차이는 무엇인가? 기본 RNN은 하나의 tanh 은닉 상태를 반복 갱신한다. LSTM은 게이트와 셀 상태를 추가한다. 같은 은닉 크기여도 파라미터 수가 다르다.
2. 128×9에서 시간축은 무엇인가? 128은 시점이고 9는 채널이다. 한 시점에서 9개 값이 들어간다.
3. 왜 마지막 출력인가? 각 2.56초 구간에 하나의 활동 정답이 있어 마지막 은닉 상태로 구간을 분류하는 기준 구조를 사용했다. 평균 집계 등은 이번 후보에 포함하지 않았다.
4. 다른 구간의 상태를 이어 쓰는가? stateful=False이며 각 구간에서 0으로 시작한다.
5. 파라미터 1,542개는 어떻게 계산하는가? 입력 288, 순환 1,024, 순환 편향 32, 출력 가중치 192, 출력 편향 6이다.
6. 전처리 완료 파일에 왜 표준화하는가? 저장 코드가 캐시에 원 시계열을 저장하고 평균·표준편차를 별도 파일로 저장한다. 한 번만 변환한다.
7. 왜 사람으로 분할하는가? 같은 사람의 중첩 구간이 양쪽에 섞이면 새 사람의 성능을 과대평가할 수 있기 때문이다.
8. 최고 seed만 쓰는가? 아니다. 2026·2027·2028 전부 평균·표본 SD로 보고하며 시연은 2026이다.
9. 최종 RNN 조건은 무엇인가? {chosen}이며 사전에 정한 세 seed 평균 validation Macro F1 규칙으로 선택했다. 기준 R01 비교표는 유지했다.
10. Dropout 비교 초기값은 같은가? R03은 같은 seed의 R02 학습 전 가중치를 복사했다. 학습 후 checkpoint는 사용하지 않았다.
11. F1과 정확도는 어떻게 다른가? Accuracy는 전체 정답 비율이고 Macro F1은 여섯 클래스의 precision과 recall 조화평균을 동등하게 평균한다.
12. 기울기 소실이 발생했는가? 시점별 gradient decay를 측정하지 않아 단정하지 않는다. 초기 고정 배치 norm은 유한성·크기 진단이다.
13. 저장 모델을 확인했는가? 별도 프로세스에서 validation 1,775개의 확률·라벨을 저장 예측과 대조하고 지표를 재계산했다.
14. 네 모델 표가 팀원 공식 결과인가? 이번 RNN 담당 작업의 같은 입력 보조 재현 실험이다. main의 과거 LSTM·MLP 결과를 그대로 복사하지 않았다.
15. 시험 성능은 얼마인가? 공식 test는 미평가이며 팀이 대상과 시점을 확정한 뒤 평가한다.
16. 가장 큰 한계는 무엇인가? 한 사람 분할, 선택에 사용한 validation, 모델 규모·튜닝 예산 차이, 정해진 착용 위치·여섯 활동 범위다.
''',encoding='utf-8')
    nb=nbformat.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    nb.cells=[
        nbformat.v4.new_markdown_cell('# 유재윤 RNN 구현과 검증\n\n실제 저장 모델과 validation만 사용한다. 공식 test는 읽지 않는다. `RNN_TRAINING_PROTOCOL_KO.md`와 결과 보고서를 함께 확인한다.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport sys, json, csv\nimport numpy as np\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom src.runtime import configure_runtime\nconfigure_runtime(2026)\nfrom src.adapters.processed_npz import load_processed_data\nfrom src.models import build_rnn\ndata = load_processed_data(ROOT/'data/processed_data.npz', ROOT/'data/preprocess.npz')\nprint('fit',data.x_fit.shape,'validation',data.x_validation.shape)\nprint('test arrays loaded:',data.provenance['test_arrays_loaded'])\nbuild_rnn().summary()"),
        nbformat.v4.new_markdown_cell('## 시간 상태와 역전파\n\n$h_t=\\tanh(x_tW_x+h_{t-1}W_h+b)$, $z=h_{128}W_o+b_o$. 구간마다 $h_0=0$이다. 작은 RNN의 수치미분은 계산 정확성을 확인하며 장기 기억 능력을 입증하지 않는다.'),
        nbformat.v4.new_code_cell("math = json.loads((ROOT/'reports/rnn/math_verification.json').read_text())\nprint(math)\nanalysis = json.loads((ROOT/'reports/rnn/analysis.json').read_text())\nfor row in analysis['summary']: print(row)\nprint('Selected RNN', analysis['selected_rnn'])"),
        nbformat.v4.new_code_cell("chosen = analysis['selected_rnn']\nrun = ROOT/'runs/rnn'/f'{chosen}_seed2026'\nfrom tensorflow import keras\nfrom src.train import predict_batches\nfrom src.training_metrics import classification_metrics, softmax\nmodel = keras.models.load_model(run/'model.keras', compile=False)\nlogits = predict_batches(model,data.x_validation)\nprint(classification_metrics(data.y_validation,logits))\ndisplay(Image(filename=str(ROOT/'reports/rnn/figures'/f'{chosen}_seed2026_history.png')))\ndisplay(Image(filename=str(ROOT/'reports/rnn/figures'/f'{chosen}_seed2026_confusion.png')))"),
        nbformat.v4.new_code_cell("prediction = logits.argmax(1)\nwrong = np.flatnonzero(prediction != data.y_validation)\nprint('validation errors',len(wrong))\nfor i in wrong[:10]: print(data.validation_ids[i],int(data.validation_subjects[i]),int(data.y_validation[i]),int(prediction[i]),float(softmax(logits[i:i+1]).max()))\nimport matplotlib.pyplot as plt\nif len(wrong):\n    bad = wrong[0]\n    good = np.flatnonzero((data.y_validation==data.y_validation[bad]) & (prediction==data.y_validation))[0]\n    fig,ax=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')\n    for panel,index,title in [(ax[0],bad,'Error'),(ax[1],good,'Correct, same true class')]:\n        panel.plot(np.arange(128)/50,data.raw_validation[index,:,0],label='body_acc_x')\n        panel.plot(np.arange(128)/50,data.raw_validation[index,:,6],label='total_acc_x')\n        panel.set(title=f'{title}: {data.validation_ids[index]} subject={data.validation_subjects[index]}',ylabel='Acceleration (g)')\n        panel.legend()\n    ax[1].set_xlabel('Time (seconds)')\n    plt.show()"),
        nbformat.v4.new_markdown_cell('## 해석 범위\n\nvalidation은 조기 종료와 구성 선택에 사용했다. seed SD는 고정 사람 분할에서의 학습 변동이다. 파형 두 사례만으로 오류 원인을 확정하지 않는다. test 평가와 다른 피험자 분할은 후속 작업이다.')
    ]
    nbformat.write(nb,ROOT/'notebooks/04_rnn_jaeyoon.ipynb')
    print(out)

if __name__=='__main__':
    main()
