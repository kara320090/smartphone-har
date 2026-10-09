# 스마트폰 행동 인식을 위한 신경망 비교 실험과 LSTM 검증

상태: 팀 취합 초안. 저자·순서·소속·교신저자·이메일·지도교수 확인 필요.

## 1. 연구 배경

스마트폰 센서 기반 행동 인식은 사용자에 따른 신호 차이를 고려해야 한다. UCI HAR의 시계열[1]에 대해 MLP·CNN·RNN·LSTM을 비교할 공통 절차를 제안하고 LSTM을 먼저 구현·검증하였다. 본 원고는 LSTM 결과만 확정된 팀 취합 초안이다.

웨어러블 행동 인식에서 CNN과 LSTM의 결합은 이미 연구되었다[4]. 본 프로젝트의 기여는 새로운 알고리즘 제안보다 사람별 분할, 전처리 연결, 반복 실험 및 저장 모델 검증을 일관된 기록으로 남기는 데 있다.

## 2. 연구 내용

입력은 128시점×9채널이며 body_acc, body_gyro, total_acc의 각 xyz 순서이다. 제공 구간에는 기존 신호 처리와 윈도잉이 적용되어 있다. 공식 train 7,352개 중 fit 5,577개·16명과 validation 1,775개·5명을 사용했다. 검증 사람은 1·6·14·21·23이다.

fit와 validation의 사람·표본 ID 중복 및 비정상 값을 검사했다. 팀의 fit 평균·표준편차를 로딩해 표준화를 한 번 적용하고, 공식 train의 0 기반 행 번호를 보존했다. 공식 test 배열과 정답은 이번 학습·선택·평가에 사용하지 않았다.

LSTM[2,3]은 sigmoid 게이트와 tanh 후보 상태를 사용하며 c_t=f_t⊙c_(t−1)+i_t⊙g_t, h_t=o_t⊙tanh(c_t)로 상태를 갱신한다. 구간별 초기 상태는 0이며 stateful=False, return_sequences=False이다[5]. 마지막 은닉 상태를 Dense 32 ReLU와 Dense 6 logits에 연결했다.

기본 L01은 은닉 크기 64와 외부 Dropout 0으로 21,222개 파라미터를 갖는다. L02는 은닉 크기 32, L03은 64와 외부 Dropout 0.3이다. 내부 Dropout은 모두 0이다. L03은 같은 seed L01의 학습 전 가중치 7개 배열을 복사·대조했다. 크기가 다른 L02는 동일 초기화 조건이 아니다.

각 조건을 seed 2026·2027·2028로 반복해 총 9회 실행했다. Adam 0.001, batch 64, 최대 40 epoch, validation loss 기준 patience 6과 최선 가중치 복원을 적용했다. 손실은 logits에 대한 희소 다중 분류 교차 엔트로피이다. 공동 비교용 LSTM 대표는 사전에 정한 L01이다.

## 3. 실험 결과 및 검증

표 1. 공동 비교의 현재 상태와 개발용 검증 결과

| 모델 | Macro F1 평균 ± SD | Accuracy 평균 |
| --- | --- | --- |
| MLP | 취합 대기 | 취합 대기 |
| CNN | 취합 대기 | 취합 대기 |
| RNN | 취합 대기 | 취합 대기 |
| LSTM L01 | 0.8813 ± 0.0195 | 0.8832 |

표 2. LSTM 통제 실험 결과와 모델 규모

| 조건 | Macro F1 평균 ± SD | 파라미터 |
| --- | --- | --- |
| L01 | 0.8813 ± 0.0195 | 21,222 |
| L02 | 0.8911 ± 0.0150 | 6,630 |
| L03 | 0.8550 ± 0.0387 | 21,222 |

표의 SD는 같은 사람 분할에서 seed 3회를 반복한 표본 SD이다. L02의 평균이 높고 L03의 평균이 낮았지만, 단일 분할과 세 seed의 관찰이므로 구조의 일반적 우월성이나 Dropout의 보편적 효과로 해석하지 않는다. L01 seed 2026의 최대 비대각 혼동은 SITTING→STANDING 148개였다.

39개 코드 검사가 통과했다. 작은 float64 LSTM의 수동 시간축 역전파를 58개 파라미터와 세 중앙차분 간격으로 검산한 최대 절대 차이는 2.48×10⁻¹⁰이었다. 망각 게이트의 sigmoid 미분을 의도적으로 누락한 대조에서는 약 0.01395의 차이를 탐지했다. 이는 실제 HAR 학습 결함이 아니다.

실제 학습은 Keras 자동미분을 사용했다. 9개 저장 모델은 새 프로세스에서 검증 1,775개 전체를 재예측해 클래스 일치와 확률 허용 오차(rtol=10⁻⁵, atol=10⁻⁶)를 확인했다. 보고 수치는 저장된 예측에서 다시 계산하였다.

## 4. 제한 조건 및 활용

validation은 epoch 선택에도 사용했으므로 독립적인 최종 test 성능이 아니다. 사람 모집단의 신뢰구간, 다른 기기·착용 위치 및 스마트폰 실시간 동작은 검증하지 않았다. 추후 모델 비교에서는 입력 표현·분할·학습 예산과 실제 구조를 함께 확인해야 한다.

동일한 데이터 규격과 예측 기반 평가를 팀 모델 취합에 활용할 수 있다. MLP가 561개 설계 특징을 쓰는 경우 128×9 시계열 비교와 구분해 보고한다. 수업 발표는 LSTM 원리·수학 검증을 중심으로, 학술 발표는 비교 조건과 결과의 범위를 중심으로 구성한다.

## 5. 결론

사람이 겹치지 않는 개발 분할에서 LSTM 구현, 9회 반복 실험 및 재로딩 검증을 완료했다. 네 모델의 상대 성능 결론은 아직 낼 수 없다. 팀원 결과와 저자·소속·지도교수를 확인하고 초록·결론을 재검토한 뒤 최종 투고본을 확정할 예정이다.

## 참고 문헌

[1] UCI Machine Learning Repository, “Human Activity Recognition Using Smartphones,” 2013. DOI: 10.24432/C54S4K.

[2] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, Vol. 9, No. 8, pp. 1735–1780, 1997.

[3] F. A. Gers, J. Schmidhuber, and F. Cummins, “Learning to Forget: Continual Prediction with LSTM,” Neural Computation, Vol. 12, No. 10, pp. 2451–2471, 2000.

[4] F. J. Ordóñez and D. Roggen, “Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition,” Sensors, Vol. 16, No. 1, 115, 2016.

[5] Keras, “LSTM layer,” https://keras.io/api/layers/recurrent_layers/lstm/ (accessed Oct. 9, 2026).
