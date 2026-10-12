# 유재윤 RNN 실험 규약

작성일 2026년 10월 12일. 아래 후보와 선택 규칙은 정식 학습 전에 고정했다. 팀 전체의 규약 확정을 대신하지 않으며, 팀에서 다른 공통 조건을 확정하면 새 실행 ID로 재실행한다.

## 데이터와 상태

제공한 `processed_data.npz`의 해시는 `00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459`다. 이 캐시는 사람 분할과 센서 전처리가 완료됐으나 배열 표준화는 적용되지 않았다. `data/preprocess.npz`의 저장 통계로 딱 한 번 표준화한다. 해당 통계가 전체 fit 배열의 평균과 표준편차와 정확히 같은지 기존 어댑터가 검증한다. 통계를 다시 fit하지 않는다.

fit 16명 5,577구간, validation 피험자 1·6·14·21·23의 1,775구간을 사용한다. 입력은 float32 `(N,128,9)`, 라벨은 int64 0–5다. 채널은 body_acc xyz, body_gyro xyz, total_acc xyz 순서다. sample_id는 `train:000000`부터 시작하는 원본 행 번호다. test 배열은 개발 학습·선택·평가 코드에서 읽지 않는다.

RNN은 `keras.layers.SimpleRNN`이다. tanh 활성화, 편향 사용, Glorot 입력 가중치, 직교 순환 가중치, 0 편향을 사용한다. `stateful=False`, `return_sequences=False`이며 각 구간의 시작 상태는 0이다. 마지막 시점 은닉 상태를 여섯 logits로 변환한다. 128시점 전체에 역전파한다. recurrent dropout과 입력 dropout은 0이다. R03의 Dropout은 순환층의 최종 출력에만 적용한다.

## 후보와 반복

| ID | 구조 | 바뀐 조건 | 파라미터 |
|---|---|---|---:|
| R01 | SimpleRNN 32 → Dense 6 | 계획서 E05에 해당하는 기준 | 1,542 |
| R02 | SimpleRNN 64 → Dense 6 | 은닉 크기 32에서 64 | 5,126 |
| R03 | SimpleRNN 64 → Dropout 0.3 → Dense 6 | R02에 출력 Dropout 추가 | 5,126 |

각 후보를 seed 2026·2027·2028에서 실행한다. R03은 같은 seed의 R02 학습 전 가중치를 복사한다. 학습 완료 가중치로 이어 학습하지 않는다. 모든 seed를 남기며 최고 seed만 보고하지 않는다.

Adam 학습률 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7, batch 64, 최대 40 epoch, SparseCategoricalCrossentropy from_logits=True를 사용한다. fit에서만 shuffle한다. val_loss 기준 patience 6, min_delta 0으로 조기 종료하고 최저 검증 손실 가중치를 복원한다. clipping은 사용하지 않는다. 연산 결정성을 켜고 CPU intra-op 4, inter-op 1을 사용한다.

최종 RNN 조건은 세 seed의 validation Macro F1 평균이 가장 높은 조건으로 선택한다. 동률이면 적은 파라미터, 이어 실험 ID 순서다. 선택에 사용한 validation 점수를 독립 최종 일반화 성능으로 표현하지 않는다. 시연 seed는 2026이다.

## 네 모델 비교

추가 비교는 동일한 팀 캐시와 통계를 사용한 M01 시계열 MLP, C01 계획서 CNN, R01 기준 RNN, 기존 main 구조 L01 LSTM을 세 seed씩 새로 실행한다. M01·C01·L01은 이번 RNN 담당 작업에서 실행한 보조 재현 실험이다. 각 담당자의 공식 수행 결과가 인계되면 데이터·분할·설정·예측을 대조한 후 별도 비교표를 만든다.

MLP 156,230개, CNN 14,054개, RNN 1,542개, main의 LSTM 64 → Dense 32 → Dense 6은 21,222개다. main의 LSTM은 계획서 LSTM 32 → Dense 6과 다른 구조다. 성능 차이를 순환 게이트 하나의 효과로 단정하지 않는다. RNN 튜닝 승자를 R01 대신 조용히 끼워 넣지 않는다. 조건별 탐색 예산이 달라 네 구조 각각의 최적 성능 비교로 표현하지 않는다.

## 확인과 저장

모든 실행은 별도 폴더에 config, split, environment와 코드 해시, history, model, preprocess, predictions, metrics, reload 기록을 저장한다. validation 1,775개 전체의 저장 예측과 새 프로세스의 예측을 rtol 1e-5, atol 1e-6으로 비교한다. 행렬은 행 정답·열 예측, Macro F1은 여섯 클래스를 모두 포함한다. SD는 seed에 대한 표본 SD ddof=1이며 피험자 모집단의 신뢰구간이 아니다.

초기 기울기 진단은 고정한 fit 첫 배치를 처음 다섯 갱신 직전에 별도로 평가한다. 진단은 training=False이고 가중치를 갱신하지 않는다. 실제 shuffle 배치의 기울기 기록이나 시간별 기울기 감쇠 측정이 아니다. 소실·폭주 또는 장기 의존성 학습의 증거로 사용하지 않는다. 작은 tanh RNN의 float64 중심차분 검사는 역전파 계산의 정확성을 별도로 확인한다.

공식 test 평가는 팀에서 대상과 시점을 동결한 다음 진행한다. 이번 개발 실험에는 포함하지 않는다.
