"""Recompute same-epoch fit/validation gaps and matched error waveform cases."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    folder=ROOT/'reports/rnn'
    a=json.loads((folder/'analysis.json').read_text())
    result=[]
    for run in a['runs']:
        hist=rows(ROOT/'runs/rnn'/run['run_id']/'history.csv')
        best=hist[run['best_epoch']-1]; last=hist[-1]
        losses=np.array([float(r['loss']) for r in hist])
        delta=np.diff(losses)
        jump=int(np.argmax(delta)) if len(delta) else 0
        jump_size=max(0.,float(delta[jump])) if len(delta) else 0.
        result.append(dict(run_id=run['run_id'],best_epoch=run['best_epoch'],
            best_fit_accuracy=float(best['accuracy']),best_validation_accuracy=float(best['val_accuracy']),
            best_fit_loss=float(best['loss']),best_validation_loss=float(best['val_loss']),
            best_accuracy_gap=float(best['accuracy'])-float(best['val_accuracy']),
            final_fit_loss=float(last['loss']),final_validation_loss=float(last['val_loss']),
            final_fit_loss_minus_best=float(last['loss'])-float(best['loss']),
            final_validation_loss_minus_best=float(last['val_loss'])-float(best['val_loss']),
            max_positive_fit_loss_jump=jump_size,jump_epoch=jump+2 if jump_size>0 else None,
            jump_previous_fit_loss=float(losses[jump]),jump_new_fit_loss=float(losses[min(jump+1,len(losses)-1)])))
    with (folder/'learning_diagnosis.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
    selected=a['selected_rnn']+'_seed2026'
    chosen=next(r for r in result if r['run_id']==selected)
    text=f'''# RNN 과적합과 학습 부족 점검

같은 epoch에서 fit·validation 지표를 비교했다. 학습 종료 시점의 fit 지표와 복원된 최선 checkpoint의 validation 지표를 섞지 않았다. 최선 epoch는 validation loss로 정했다. 아래 값의 원자료는 각 실행 history.csv이며 재생성 명령은 `python reports/rnn/reproduce_diagnosis.py`다.

선택 조건 {selected}의 최선 epoch {chosen['best_epoch']}에서 fit Accuracy는 {chosen['best_fit_accuracy']:.4f}, validation Accuracy는 {chosen['best_validation_accuracy']:.4f}였다. 차이는 {chosen['best_accuracy_gap']*100:.2f} percentage points다. 같은 시점의 fit loss는 {chosen['best_fit_loss']:.4f}, validation loss는 {chosen['best_validation_loss']:.4f}였다.

| 실행 | 최선 epoch | fit Accuracy | val Accuracy | fit loss | val loss | 마지막 fit loss 변화 | 마지막 val loss 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in result:
        if r['run_id'].startswith('R'):
            text+=f"| {r['run_id']} | {r['best_epoch']} | {r['best_fit_accuracy']:.4f} | {r['best_validation_accuracy']:.4f} | {r['best_fit_loss']:.4f} | {r['best_validation_loss']:.4f} | {r['final_fit_loss_minus_best']:+.4f} | {r['final_validation_loss_minus_best']:+.4f} |\n"
    text+='''
마지막 loss 변화는 마지막 epoch에서 최선 epoch 값을 뺀 것이다. fit loss가 감소하면서 validation loss가 증가한 실행은 후반 과적합과 양립하는 관찰이다. 다만 loss 변동과 데이터 분포 차이도 가능한 설명이므로 이 두 수치만으로 원인을 확정하지 않는다. 조기 종료는 최저 validation loss의 가중치를 복원해 후반 악화를 결과 모델에 그대로 사용하지 않도록 했다.

R01의 낮은 validation 점수만으로 과소적합을 판정하지 않았다. 학습 정확도·손실도 함께 기록했으며 은닉 크기와 Dropout을 바꾼 조건에서 개선의 일관성을 세 seed로 확인했다. 한 seed의 개선이나 더 큰 모델의 점수만으로 용량 부족이 원인이라고 단정할 수 없다. 학습률·출력 집계·시점별 기울기 등을 분리한 추가 대조는 이번 후보에 포함되지 않았다.

학습 중 loss는 Dropout 적용 상태에서 계산되고 validation은 평가 모드에서 계산된다. 따라서 R03의 두 손실 차이에는 모드 차이도 포함된다. 초기 기울기 probe는 고정 fit 배치의 별도 진단이며 실제 갱신 배치의 gradient norm이나 시점별 소실 측정이 아니다.
'''
    unstable=max((r for r in result if r['run_id'].startswith('R')),key=lambda r:r['max_positive_fit_loss_jump'])
    if unstable['max_positive_fit_loss_jump']>0:
        text+=f"\n학습 loss의 인접 epoch 증가를 따로 확인했다. 가장 큰 RNN 증가 사례는 {unstable['run_id']}의 epoch {unstable['jump_epoch']}이며, 직전 {unstable['jump_previous_fit_loss']:.4f}에서 {unstable['jump_new_fit_loss']:.4f}로 증가했다. 이는 학습 곡선에서 확인한 변동이다. 해당 시점의 실제 batch 기울기를 저장하지 않아 이를 기울기 폭주로 확정하지 않는다. 모든 실행의 최대 증가량과 시점은 learning_diagnosis.csv에 있다.\n"
    (ROOT/'docs/rnn-report/LEARNING_DIAGNOSIS_KO.md').write_text(text,encoding='utf-8')
    from src.adapters.processed_npz import load_processed_data
    data=load_processed_data(ROOT/'data/processed_data.npz',ROOT/'data/preprocess.npz')
    pred=rows(ROOT/'runs/rnn'/selected/'validation_predictions.csv')
    p=np.array([int(r['predicted_label']) for r in pred])
    cm=np.array(json.loads((folder/f'{selected}_confusion.json').read_text())['counts'])
    off=cm.copy(); np.fill_diagonal(off,0)
    truth,wrong_class=np.unravel_index(off.argmax(),off.shape)
    wrong=np.flatnonzero((data.y_validation==truth)&(p==wrong_class))[:3]
    names=['WALKING','UPSTAIRS','DOWNSTAIRS','SITTING','STANDING','LAYING']
    cases=[]; arrays={}; used_success=set()
    for number,index in enumerate(wrong,1):
        remaining=np.ones(len(p),dtype=bool)
        remaining[list(used_success)]=False
        success=np.flatnonzero((data.y_validation==truth)&(p==truth)&(data.validation_subjects==data.validation_subjects[index])&remaining)
        matched='same subject and true class'
        if not len(success):
            success=np.flatnonzero((data.y_validation==truth)&(p==truth)&remaining); matched='same true class, subject can differ'
        if not len(success):
            continue
        good=int(success[0]); index=int(index)
        used_success.add(good)
        record=dict(pair=number,error_sample_id=str(data.validation_ids[index]),success_sample_id=str(data.validation_ids[good]),
            error_subject=int(data.validation_subjects[index]),success_subject=int(data.validation_subjects[good]),
            true_class=int(truth),error_predicted_class=int(wrong_class),matching=matched)
        cases.append(record)
        arrays[f'error_{number}']=data.raw_validation[index]; arrays[f'success_{number}']=data.raw_validation[good]
        fig,axes=plt.subplots(3,2,figsize=(10,7),sharex=True,layout='constrained')
        for column,row in enumerate((index,good)):
            for group in range(3):
                for axis,name in enumerate('xyz'):
                    axes[group,column].plot(np.arange(128)/50,data.raw_validation[row,:,3*group+axis],label=name,lw=1)
                axes[group,column].set_ylabel(['Body acc (g)','Body gyro (rad/s)','Total acc (g)'][group])
                axes[group,column].legend(ncol=3,fontsize=8)
            axes[0,column].set_title(f"{'Error' if column==0 else 'Correct'} {data.validation_ids[row]} subject={data.validation_subjects[row]}\ntrue={names[truth]}, predicted={names[p[row]]}")
            axes[-1,column].set_xlabel('Time (seconds)')
        fig.savefig(folder/'figures'/f'error_success_pair_{number}.png',dpi=150)
        plt.close(fig)
    np.savez_compressed(folder/'error_success_waveforms.npz',**arrays)
    (folder/'error_success_cases.json').write_text(json.dumps(dict(run_id=selected,selection='first three original IDs in largest directed confusion; equal number of correctly classified matches',cases=cases),indent=2),encoding='utf-8')
    text='# RNN 오류와 성공 파형 대조\n\n가장 큰 방향성 혼동에서 원본 행 번호가 빠른 오분류 세 개와, 같은 정답 클래스의 성공 사례를 같은 수로 선택했다. 가능한 경우 같은 사람을 우선했다. 사례는 무작위 대표 표본이 아니며 파형만으로 원인을 확정하지 않는다. 시계열은 표준화 전 값이고 가로축은 초, 세로축은 가속도 g와 각속도 rad/s다. 재생성 명령은 `python reports/rnn/reproduce_diagnosis.py`다.\n\n'
    for r in cases:
        text+=f"오류 `{r['error_sample_id']}`와 성공 `{r['success_sample_id']}`의 비교 조건은 {r['matching']}다.\n\n![오류 성공 대조 {r['pair']}](../../reports/rnn/figures/error_success_pair_{r['pair']}.png)\n\n"
    (ROOT/'docs/rnn-report/ERROR_CASES_KO.md').write_text(text,encoding='utf-8')
    summary={r['experiment']:r for r in a['summary']}
    rnn=summary['R01']; tuned=summary[a['selected_rnn']]
    comparison='''# 네 모델 성능 비교 분석

같은 전처리 NPZ·저장 통계·S0 사람 분할·세 seed를 사용해 재실행했다. MLP·CNN·LSTM 행은 이번 RNN 담당 작업에서 실행한 비교용 보조 실험이다. 다른 담당자의 공식 제출 결과를 대신하지 않는다. validation은 학습 종료와 checkpoint 선택에 사용했다. 원자료는 four_model_comparison.csv와 paired_differences.csv다.

| 모델과 조건 | 파라미터 | Accuracy 평균 | Macro F1 평균 ± seed SD |
|---|---:|---:|---:|
'''
    for exp in ('M01','C01','R01','L01'):
        r=summary[exp]
        comparison+=f"| {r['model']} {exp} | {r['parameters']:,} | {r['accuracy_mean']:.4f} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} |\n"
    difference=summary['L01']['macro_f1_mean']-summary['C01']['macro_f1_mean']
    comparison+=f"\nLSTM의 평균 Macro F1은 CNN보다 {difference:.4f} 높았다. 세 번의 학습과 한 고정 validation 분할에서의 평균 차이이며 통계적 우월성이나 다른 사람 분할에서도 유지될 차이로 표현하지 않는다. LSTM과 CNN은 모델 규모·연산·출력 집계도 다르다.\n\n"
    comparison+='| seed | MLP Macro F1 | CNN Macro F1 | RNN Macro F1 | LSTM Macro F1 |\n|---|---:|---:|---:|---:|\n'
    for seed in (2026,2027,2028):
        values=[next(r['macro_f1'] for r in a['runs'] if r['experiment']==exp and r['seed']==seed) for exp in ('M01','C01','R01','L01')]
        comparison+='| '+str(seed)+' | '+' | '.join(f'{v:.4f}' for v in values)+' |\n'
    comparison+=f"\n기준 RNN의 평균은 {rnn['macro_f1_mean']:.4f}, seed SD는 {rnn['macro_f1_sd']:.4f}였다. 이 실험에서는 다른 세 기준 구성보다 평균이 낮고 seed 변동이 컸다. RNN이라는 구조 전체의 한계나 기울기 소실의 발생을 의미하지 않는다. 기준의 파라미터가 1,542개이고 마지막 시점 출력만 사용한 조건도 함께 고려해야 한다.\n\n"
    comparison+=f"별도 튜닝에서 선택한 {a['selected_rnn']}의 평균 Macro F1은 {tuned['macro_f1_mean']:.4f}로 R01보다 {(tuned['macro_f1_mean']-rnn['macro_f1_mean'])*100:.2f} percentage points 높았다. 튜닝 조건은 동일 분할의 validation으로 선택했으므로 기준 네 모델 비교표에서 R01을 대체하지 않았다. 네 모델 모두 같은 탐색 예산으로 최적화한 결과가 아니다.\n\n"
    comparison+='학습 시간은 run별 train_seconds로 보관했다. epoch 수가 조기 종료로 달라지고 RNN의 추가 gradient 진단 callback도 포함되어 있어 그 숫자를 순수 구조의 속도 순위로 해석하지 않는다. 추론 속도는 이번 비교에서 별도로 측정하지 않았다. 최종 test와 추가 사람 분할 검증은 후속 작업이다.\n'
    (ROOT/'docs/rnn-report/FOUR_MODEL_COMPARISON_KO.md').write_text(comparison,encoding='utf-8')
    print('Learning diagnosis and',len(cases),'matched waveform pairs created')

if __name__=='__main__':
    main()
