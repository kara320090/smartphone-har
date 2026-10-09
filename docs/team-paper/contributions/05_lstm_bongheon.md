# 이봉헌 LSTM 논문 기여 원고

본 연구의 LSTM은 동일한 스마트폰 센서 구간을 순차적으로 처리하는 비교 모델이다. 전처리가 준비된 UCI HAR 자료에서 128시점과 9채널의 입력을 사용하고, 학습 사람 16명과 검증 사람 5명을 분리하였다. 저장된 학습 통계로 표준화를 한 번 적용했으며 공식 test는 개발에 사용하지 않았다.

모델은 64개 은닉 유닛의 LSTM, 32차원 ReLU Dense, 여섯 logits의 출력층으로 구성하였다. 구간마다 상태를 독립적으로 초기화하고 마지막 은닉 상태로 활동을 분류하였다. 총 파라미터 수는 21,222개다. 손실은 logits 기반 sparse categorical cross-entropy로 정의하였다.

Adam 학습률 0.001, batch 64, 최대 40 epoch, 검증 손실 기준 patience 6을 공통 적용하였다. seed 2026·2027·2028에서 반복하고 검증 손실이 최소인 가중치를 복원하였다. 모델과 전처리·표본 ID·설정·환경을 같은 실행 단위로 저장하였다.

기준 LSTM의 검증 Macro F1은 0.8813 ± 0.0195, Accuracy 평균은 88.32%였다. 표준편차는 고정 분할에서 세 학습 seed의 변동이다. 각 실행의 예측에서 지표를 재계산하고, 새 프로세스에서 1,775개 검증 구간의 저장 모델 예측을 대조하였다.

추가 분석에서는 은닉 크기 32의 모델과 외부 Dropout 0.3 조건을 비교하였다. 두 조건의 Macro F1 평균은 각각 0.8911과 0.8550였다. Dropout 조건은 같은 seed의 기본 모델 학습 전 가중치를 복사하였다. 보조 조건은 공동 모델 비교용 기준 결과와 분리하였다.

작은 LSTM의 NumPy 순전파와 시간축 역전파를 자동미분 및 중앙차분과 비교하여 계산을 검산하였다. 이 교육용 검사와 실제 Keras 기반 학습을 구분하였다. 단일 검증 사람 분할과 적은 seed 수의 한계가 있으며, 다른 팀원 모델의 공정 비교와 공식 test 평가는 공동 후속 절차에 따른다.

## 참고문헌

[1] UCI Machine Learning Repository. Human Activity Recognition Using Smartphones. DOI 10.24432/C54S4K. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

[2] Anguita D. et al. A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN, 2013. https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf

[3] Hochreiter S., Schmidhuber J. Long Short-Term Memory. Neural Computation 9(8), 1735–1780, 1997. https://www.bioinf.jku.at/publications/older/2604.pdf

[4] Gers F. A., Schmidhuber J., Cummins F. Learning to Forget: Continual Prediction with LSTM. Neural Computation 12(10), 2451–2471, 2000. https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf

[5] Keras 3. LSTM layer. https://keras.io/api/layers/recurrent_layers/lstm/ (확인 2026-10-09).

[6] Ordóñez F. J., Roggen D. Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. Sensors 16(1), 115, 2016. DOI 10.3390/s16010115. https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/

## 구조표

| 층 | 출력 | L01 파라미터 |
|---|---|---:|
| 입력 | B × 128 × 9 | 0 |
| LSTM 64 | B × 64 | 18,944 |
| Dense ReLU | B × 32 | 2,080 |
| 외부 Dropout 0 | B × 32 | 0 |
| Dense logits | B × 6 | 198 |
| 총합 | 6클래스 | 21,222 |

## 공동 비교에 제공할 L01 결과

| seed | validation Accuracy | validation Macro F1 | 선택/실행 epoch |
|---:|---:|---:|---:|
| 2026 | 0.8614 | 0.8590 | 4/10 |
| 2027 | 0.8986 | 0.8952 | 3/9 |
| 2028 | 0.8896 | 0.8896 | 6/12 |

Macro F1 평균 ± 표본 SD: **0.8813 ± 0.0195**. Accuracy 평균 ± 표본 SD: **0.8832 ± 0.0194**.

## 보조 조건의 결과

| 조건 | Macro F1 평균 | seed 표본 SD | 파라미터 |
|---|---:|---:|---:|
| L01 | 0.8813 | 0.0195 | 21,222 |
| L02 | 0.8911 | 0.0150 | 6,630 |
| L03 | 0.8550 | 0.0387 | 21,222 |

공식 test는 미평가다. L02·L03은 개인 보조 분석이며 네 모델의 공동 비교 대표를 결과 후 교체하는 근거로 사용하지 않는다. 다른 팀원 모델의 결과를 측정한 원고가 아니므로 MLP·CNN·RNN의 상대 성능은 후속 취합 때 작성한다.

근거: reports/lstm/summary.csv, runs.csv, initialization_audit.json, dataset_audit.json, 각 실행의 fresh_process_verification.json. 학습 코드 고정 커밋은 95422e0993935be59db21248742f83c5fdf5453d다.
