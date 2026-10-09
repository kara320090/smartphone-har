# 스마트폰 센서 행동 인식 모델 비교

공동 논문 취합용 초안  2026년 10월 9일  취합 이봉헌

상태: 데이터 확인과 LSTM 실험은 완료했다. MLP·CNN·SimpleRNN 결과, 최종 test 평가, 저자 순서·소속은 미확정이다. 이 문서는 내부 검토용이며 대학생 공식 2단 1~2쪽 제출본은 아니다.

## 초록 초안

본 연구는 UCI HAR의 센서 구간을 사용하여 MLP, 1D CNN, SimpleRNN과 LSTM을 동일한 사람 분할에서 비교하는 실험을 설계한다. 현재 LSTM 기준 모델의 개발 validation Macro F1은 세 학습 seed에서 0.8813 ± 0.0195이며, 이는 고정 분할의 학습 변동이다. 나머지 모델의 결과를 취합한 뒤 모델 간 차이와 최종 평가를 포함해 초록을 완성한다. 현 단계의 결과로 최우수 모델을 결론 내리지 않는다.

## 서론과 관련 연구

스마트폰의 가속도·각속도 구간으로 사용자의 활동을 분류하는 문제는 공개 HAR 데이터로 연구되어 왔다[1,2]. 본 프로젝트는 시간축을 펼치는 MLP, 지역 패턴을 처리하는 1D CNN, 순차 상태를 갱신하는 SimpleRNN과 LSTM의 차이를 같은 입력과 평가 규약 아래 살펴본다. 비교 질문은 고정된 검증 사람에서 구조별 성능, 학습 seed 변동과 모델 규모가 어떻게 다른가이다.

LSTM의 셀 상태와 게이트 구조는 기존 연구의 방법이다[3,4]. CNN과 LSTM을 결합한 웨어러블 활동 인식 연구도 존재한다[6]. 이번 연구의 기여 후보는 데이터·초기화·평가 절차를 확인한 비교 근거와 오류 분석이다. 새로운 LSTM 알고리즘 또는 HAR 최초 비교로 표현하지 않는다. 관련 연구의 정확한 비교 범위와 최종 기여 문장은 팀 결과 취합 후 검토한다.

## 데이터와 분할

UCI HAR는 허리에 착용한 스마트폰으로 관측한 여섯 활동의 자료이며, 제공 구간은 이미 신호 처리와 윈도잉을 거쳤다[1,2]. 팀 캐시의 공식 train 7,352개 구간 중 fit 5,577개·16명, validation 1,775개·5명을 사용한다. 검증 사람은 1·6·14·21·23이다. 입력은 float32의 128시점×9채널이며 body_acc xyz, body_gyro xyz, total_acc xyz 순서다. fit 통계 파일을 로딩해 한 번 표준화하며 표본 ID는 공식 train의 0 기반 원 행 번호를 보존한다.

# 모델과 실험 조건

## 팀 모델 구조 취합

| 모델 | 담당 | 입력과 구조 | 파라미터 |
| --- | --- | --- | --- |
| MLP | 한윤섭 | 같은 시계열을 펼친 구조 제안
실제 구조 확인 대기 | 확인 대기 |
| 1D CNN | 최승빈 | 시간축 Conv1D
커널·필터·집계 확인 대기 | 확인 대기 |
| SimpleRNN | 유재윤 | 실제 유닛·상태·출력 확인 대기 | 확인 대기 |
| LSTM L01 | 이봉헌 | LSTM 64, Dense 32 ReLU
외부 Dropout 0, Dense 6 logits | 21,222 |

MLP가 561개 설계 특징을 사용한다면 128×9 시계열 비교와 입력 표현이 다르므로 별도 조건으로 표시한다. 같은 은닉 크기라도 RNN과 LSTM의 파라미터 수는 같지 않다. 입력 표현, 모델 규모와 탐색 예산 차이를 구조 자체의 효과로 단정하지 않는다.

## LSTM 구현

LSTM은 tanh 후보 상태와 sigmoid 게이트를 사용하고[3–5], 구간마다 은닉·셀 상태를 0으로 초기화한다. stateful=False, return_sequences=False로 마지막 은닉 상태를 분류기에 전달한다. LSTM 내부 Dropout과 recurrent Dropout은 0이다. 손실은 SparseCategoricalCrossentropy(from_logits=True)이며 최종 Dense 층은 softmax를 포함하지 않는다.

`c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t; h_t = o_t ⊙ tanh(c_t)`

## 확인된 LSTM 조건과 공동 적용 제안

| 항목 | LSTM 실제 수행 조건 |
| --- | --- |
| seed | 2026, 2027, 2028 |
| 학습 | Adam 0.001, batch 64, 최대 40 epoch |
| 선택 | val_loss 최소, patience 6, 최선 가중치 복원 |
| 평가 | 같은 validation 1,775개, Accuracy와 6클래스 Macro F1 |
| 통제 조건 | L01 H64 p0 / L02 H32 p0 / L03 H64 p0.3 |

L03은 같은 seed L01의 학습 전 가중치 7개 배열을 복사·대조했다. L02는 크기가 달라 동일 초기화 비교가 아니다. 나머지 모델의 조건은 기록 확인 대기다.

# 확인된 결과와 분석

## 공동 비교표의 현재 상태

| 모델 | 반복 | validation Macro F1
평균 ± seed 표본 SD | Accuracy 평균 |
| --- | --- | --- | --- |
| MLP | 확인 대기 | 결과 취합 대기 | 결과 취합 대기 |
| 1D CNN | 확인 대기 | 결과 취합 대기 | 결과 취합 대기 |
| SimpleRNN | 확인 대기 | 결과 취합 대기 | 결과 취합 대기 |
| LSTM L01 | 3 | 0.8813 ± 0.0195 | 0.8832 |

표의 비어 있는 결과는 0이나 미측정 성공값으로 대체하지 않는다. LSTM 값은 저장된 예측에서 재계산한 지표다. seed SD는 같은 사람 분할에서 학습을 반복했을 때의 변동이며 사람 모집단 신뢰구간이 아니다. validation은 epoch 선택에도 사용했으므로 독립된 최종 test 성능과 구분한다.

## LSTM 개인 보조 분석

| 조건 | Macro F1 평균 | seed 표본 SD | 파라미터 |
| --- | --- | --- | --- |
| L01 | 0.8813 | 0.0195 | 21,222 |
| L02 | 0.8911 | 0.0150 | 6,630 |
| L03 | 0.8550 | 0.0387 | 21,222 |

소형 L02의 평균이 기준보다 높고 외부 Dropout L03의 평균이 낮았지만, 단일 사람 분할·세 seed의 관찰이다. 보편적인 구조 우월성이나 Dropout의 일반적 효과를 결론 내리지 않는다. 공동 비교용 LSTM 대표는 결과 전에 정한 L01으로 유지한다. 대표 실행 L01 seed 2026에서 가장 큰 비대각 혼동은 SITTING을 STANDING으로 예측한 148개 구간이다. 원인을 센서 위치나 개인 습관으로 확정하지 않는다.

## 구현과 저장 검증

39개 코드 검사가 통과했고, 작은 float64 LSTM의 58개 파라미터를 세 중앙차분 간격으로 검산했다. 중앙차분과 수동 역전파의 최대 절대 차이는 2.48×10⁻¹⁰이다. 망각 게이트의 sigmoid 미분을 의도적으로 누락한 대조는 자동미분과 약 0.01395 차이를 보여 검사가 탐지했다. 이 대조는 실제 HAR 학습 결함이 아니다. 실제 학습은 Keras 자동미분을 사용하며, 9개 저장 모델은 새 프로세스에서 validation 전체를 재예측해 원 예측과 대조했다.

# 논의와 취합 후 완료할 사항

## 현재 결론의 범위

현재 결과는 고정된 사람 분할에서 LSTM을 구현하고 반복 평가한 근거를 제공한다. 네 모델의 상대 성능과 최종 일반화 결론은 아직 작성할 수 없다. 더 넓은 사용자 집단, 다른 기기나 착용 위치의 성능은 검증하지 않았다. CPU 추론 시간은 PC의 batch 1, 준비된 입력에서 모델 호출과 출력 반환까지의 측정이며 스마트폰 실장치 성능이 아니다.

## 원고 완성 절차

정회서는 데이터·전처리 설명을 확인하고, 한윤섭·최승빈·유재윤은 실제 구조와 세 seed 예측·설정·개인 원고를 제출한다. 이봉헌은 L01 결과를 포함해 공통 지표를 취합한다. 모든 담당자가 자기 수치와 해석을 확인한 뒤 저자·소속·순서·발표자를 확정한다. 공식 test 평가 시점과 선택 고정 절차는 팀·지도교수와 합의한다. 그 후 결과·초록·결론을 완성하고 대학생 공식 2단 1~2쪽 양식으로 옮긴다. 공식 공고는 대학생 포스터 발표를 안내한다[7].

수업 발표와 학술 발표는 같은 연구를 공유한다. 실제 작성·수정에 Codex 보조를 사용했으며 계산·코드·결과의 확인 근거를 남겼다. 투고본의 인용·저자 책임·AI 사용 표기는 학회 요구와 지도교수 검토를 따른다. 현 단계는 최종 유사도 검사나 모든 선행 연구와의 신규성 검토를 완료한 상태가 아니다.

## 참고문헌

[1] UCI Machine Learning Repository. Human Activity Recognition Using Smartphones. DOI 10.24432/C54S4K. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

[2] Anguita D. et al. A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN, 2013. https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf

[3] Hochreiter S., Schmidhuber J. Long Short-Term Memory. Neural Computation 9(8), 1735–1780, 1997. https://www.bioinf.jku.at/publications/older/2604.pdf

[4] Gers F. A., Schmidhuber J., Cummins F. Learning to Forget: Continual Prediction with LSTM. Neural Computation 12(10), 2451–2471, 2000. https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf

[5] Keras. LSTM layer. https://keras.io/api/layers/recurrent_layers/lstm/ (확인 2026-10-09).

[6] Ordóñez F. J., Roggen D. Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. Sensors 16(1), 115, 2016. DOI 10.3390/s16010115.

[7] 한국디지털콘텐츠학회. 2026 추계종합학술대회 및 대학생논문경진대회 모집 안내. https://dcs.or.kr/homepage/custom/submit (확인 2026-10-09).
