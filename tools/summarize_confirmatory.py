"""Human-readable report from audited results; never trains or selects a model."""
from pathlib import Path
import csv,json,statistics
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'reports/confirmatory_20261019'
OUT=ROOT/'docs/dcs-team-paper-research-v10'
OUT.mkdir(parents=True,exist_ok=True)
def read(name):
    with (REP/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
audit=json.loads((REP/'AUDIT.json').read_text());assert audit['passed']
summary=read('summary.csv');runs=read('test_runs.csv');bench=read('benchmark_runs.csv');persons=read('test_subjects.csv')
idx={(r['condition'],r['level']):r for r in summary}
def fs(r):return f"{float(r['f1_mean']):.4f}"+(f" ± {float(r['f1_seed_sd']):.4f}" if r['f1_seed_sd'] else '')
lines=['# v10 추가 실험과 공식 test 평가 상세 결과','',
       '공식 test 9명·2,947구간. 학습 16명·5,577구간, epoch 선택 검증 5명·1,775구간. 기존 자료를 본 뒤 설계한 후속 실험이다. 모든 평가 모델·코드·통계·개발 결과를 고정한 뒤 test를 평가했다.','',
       '## 1. 고정한 비교와 결과','',
       '| 조건 | 평가 수준 | 학습 반복 | 파라미터 | 공식 test Macro F1 | Accuracy 평균 |',
       '|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['condition']} | {r['level']} | {r['runs']} | {int(r['parameters']):,} | {fs(r)} | {float(r['accuracy_mean']):.4f} |")
lines+=['','NN은 세 seed 평균±표본 SD, LR은 단일 학습이다. 같은 test 사람들에서 세 번 평가했으므로 SD를 사람 모집단의 신뢰구간으로 읽지 않는다. 모든 seed를 포함하며 최고 test 점수의 모델을 골라 대표로 삼지 않았다.','',
        '## 2. 입력과 목표를 맞춘 비교','']
lr=idx['LR3','three'];fm=idx['FMLP3','three'];delta=float(fm['f1_mean'])-float(lr['f1_mean'])
lines += [f"LR3와 FMLP3는 같은 18개 특징과 직접 3상태 정답을 사용한다. LR3 F1={fs(lr)}, FMLP3 F1={fs(fm)}, 차이(FMLP3−LR3)={delta:+.6f}이다. 이는 지정한 두 구성의 비교이며 통계적 동등성·비열등성 검정을 한 결과가 아니다.",
          '', 'FMLP3는 18→128→64→3 ReLU MLP이며 파라미터 10,883개다. LR3는 선형 logits의 가중치·편향 57개다. 입력 특징을 맞췄지만 최적화법·분류기 용량까지 같지는 않다. LR 입력은 float64, MLP 입력은 float32이다.','',
          '## 3. 6종 예측 병합과 직접 3상태 학습','',
          '| 모델 | 6종 F1 | 6종→3상태 F1 | 직접 3상태 F1 |','|---|---:|---:|---:|']
for name,old,new in [('MLP','M01','MLP3'),('CNN','C01','CNN3'),('RNN','R01','RNN3'),('LSTM','L01','LSTM3'),('LR','LR6','LR3')]:
    lines.append(f'| {name} | {fs(idx[old,"six"])} | {fs(idx[old,"three"])} | {fs(idx[new,"three"])} |')
lines+=['','같은 seed의 직접학습−병합 차이는 `paired_differences.csv`에 모두 남겼다. 두 과제의 F1 차이를 원래 6종 과제의 정확도 개선으로 표현하지 않는다. 신경망은 시계열 입력, LR는 통계 특징을 쓰므로 LR–시계열 모델 비교는 전체 구성의 비교다.','',
        '## 4. 비용 측정','',
        '| 직접 3상태 조건 | 구간→출력 ms | 가중치+전처리 배열 bytes | 저장 모델 bytes 범위 | 프로세스 RSS MiB 범위 |','|---|---:|---:|---:|---:|']
for c in ['LR3','FMLP3','MLP3','CNN3','RNN3','LSTM3']:
    r=idx[c,'three'];b=[q for q in bench if q['condition']==c]
    sizes=[int(q['stored_model_bytes']) for q in b];rss=[int(q['sampled_peak_rss_bytes'])/2**20 for q in b]
    lines.append(f"| {c} | {float(r['window_ms_median_across_runs']):.4f} | {r['weight_and_preprocessing_bytes']} | {min(sizes):,}–{max(sizes):,} | {min(rss):.1f}–{max(rss):.1f} |")
env=json.loads((REP/'benchmark/LR3.json').read_text())['environment']
lines+=['',f"환경: {env['processor']}; {env['platform']}; TensorFlow {env['tensorflow']}, Keras {env['keras']}. CPU intra-op 4 / inter-op 1. batch 1, 워밍업 50회, 각 실행 500회. 표의 시간은 실행별 중앙값의 중앙값이다. 각 실행 p95와 원시 500회 기록도 공개했다.",
        '', '센서 구간→출력은 표준화, 필요한 특징 계산, 모델 호출, softmax와 argmax의 결과 반환을 포함한다. 이미 UCI에서 필터링·구간화한 센서 창을 시작점으로 하므로 물리 센서의 첫 입력부터의 지연이 아니다. 입력 수집·필터링·구간 생성·파일 읽기·모델 로딩·최초 tracing은 제외한다. 준비 입력 이후 호출 시간은 `benchmark_runs.csv`의 별도 열에 있다.',
        '', '배열 바이트는 가중치·편향과 팀 표준화 상수, 해당하는 경우 특징 표준화 상수의 수치 저장량이다. Python 객체·중간 activation·런타임·특징 추출의 작업 메모리는 제외한다. 저장 모델은 포맷 오버헤드를 포함하며 기존 6종 모델 파일에는 학습 optimizer 정보가 포함될 수 있다. RSS는 런타임까지 포함한 전체 프로세스를 매 반복 후 관측한 최대치다. 순간 최대 할당량이나 모델 전용 메모리가 아니다.',
        '', 'LR은 NumPy, NN은 TensorFlow tf.function이므로 속도 차이에는 런타임 구현도 포함된다. 이 PC 결과를 스마트폰 속도·전력 절감으로 옮겨 쓰지 않는다.','',
        '## 5. 사람별 결과','',
        '| 사람 | 구간 수 | LR3 F1 | FMLP3 F1 평균 | MLP3 | CNN3 | RNN3 | LSTM3 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for sid in sorted(set(int(r['subject']) for r in persons)):
    vals=[];count=None
    for c in ['LR3','FMLP3','MLP3','CNN3','RNN3','LSTM3']:
        a=[r for r in persons if r['run_id'].split('_seed')[0]==c and int(r['subject'])==sid and r['level']=='three']
        assert a
        count=a[0]['count'];vals.append(f"{statistics.mean(float(r['macro_f1']) for r in a):.4f}")
    lines.append(f'| {sid} | {count} | '+' | '.join(vals)+' |')
lines+=['','각 사람은 여러 중첩 구간을 제공하므로 구간을 모두 독립 관측으로 취급하지 않는다. 사람별 결과는 분포를 보여 주는 기술 통계다.','',
        '## 6. 관찰한 오분류 예시','']
names=['걷기','계단 오르기','계단 내리기','앉기','서기','눕기'];group=['이동','앉거나 서 있음','누워 있음']
for runid in ['LR3','FMLP3_seed2026','LSTM3_seed2026']:
    saved=read(f'test/{runid}/predictions.csv');bad=[r for r in saved if r['prediction']!=r['true_target']]
    lines += [f'### {runid}: {len(bad)}/2,947개 오류','']
    if not bad:lines+=['이 실행의 공식 test에서 오분류가 없었다. 일상생활 전체의 무오류를 뜻하지 않는다.',''];continue
    lines+=['| 원 행 ID | 사람 | 실제 6종 | 예측 요약 |','|---|---:|---|---|']
    for r in bad[:8]:lines.append(f"| {r['sample_id']} | {r['subject']} | {names[int(r['true_six'])]} | {group[int(r['prediction'])]} |")
    lines+=['','원 행 순서의 첫 최대 8개 사례다. 센서 구간을 직접 추가 분석하지 않은 상태에서 오분류 원인을 단정하지 않는다.','']
lines+=['## 7. 재현성 및 남은 한계','',
        f"저장 예측 {audit['test_predictions']:,}행, 지표 재계산 {audit['metric_recomputations']}건, 새 프로세스 전체 test 재예측 {audit['fresh_process_replays']}개 모델을 확인했다. 원본 UCI 센서값·라벨·사람과 캐시의 행 정합성을 대조했다. 모델·코드·통계는 FINAL_LOCK.json의 해시와 일치한다.",
        '', '조건을 개발 단계에서 정했더라도 공개된 한 데이터셋의 한 공식 test라는 제한은 남는다. 다양한 휴대 위치, 다른 기기, 전환 동작, 연속 일상 신호, 전력 소비와 앱 효용은 확인하지 않았다. 누워 있음은 수면이 아니며 활동 상태 분류만으로 건강 상태를 판정하지 않는다.',
        '', '2페이지 원고는 핵심 표·그림으로 요약하고, 모든 실행과 메모리 측정·사람별 성능·오류 사례는 이 보충자료와 CSV에 보존한다. 연구 결과에 맞춰 활용 가능성을 제안하며, 앱을 구현한 것처럼 기술하지 않는다.','']
(OUT/'DETAILED_RESULTS_KO.md').write_text('\n'.join(lines),encoding='utf-8')
print('Detailed report written from audited CSV/JSON')
