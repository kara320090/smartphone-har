# 이봉헌 개인 발표 원고와 예상 질문

PPT는 본문 12장과 보충자료 5장으로 구성했습니다. 전체 설명은 약 8–10분을 기준으로 하며 실제 말하는 속도에 따라 달라집니다.

## 4분 축약 발표

1 → 3 → 4 → 6 → 7 → 12번 슬라이드 순서입니다. 목표 시간은 15 + 40 + 40 + 45 + 55 + 45 = 240초입니다. 해당 노트의 [4분 경로] 문단을 연습하고 질문에 따라 보충자료를 활용하세요.

## 슬라이드 1

[전체 발표 약 8–10분 / 4분 축약 경로: 1→3→4→6→7→12]
[4분 경로 15초] 안녕하세요. 기준모델 구현과 공통 학습을 맡은 이봉헌입니다. MLP를 구현한 뒤 역전파를 확인하고, 동일한 사람 분할에서 일곱 가지 조건을 비교했습니다. 오늘 설명하는 값은 공식 시험 점수가 아닌 검증 결과입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/BONGHEON_WORK_SUMMARY.md

## 슬라이드 2

제 역할의 중심은 기준모델과 공통 학습 코드입니다. 전처리 담당자의 작업과 독립적으로 진행하도록 UCI 참조 어댑터를 만들었고, 이후 팀 공식 모듈과 교체할 수 있게 입력 계약을 정의했습니다. CNN·RNN·LSTM·AE, 전체 시험 평가, 시연 시스템의 완료를 제 성과로 포함하지 않습니다. 실제 팀원 환경의 실행 확인도 남아 있습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/train.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/HANDOFF_KO.md

## 슬라이드 3

[4분 경로 40초] UCI는 공식적으로 사람 기준 train과 test를 제공합니다. 저는 공식 train 안에서 다시 사람 1, 6, 14, 21, 23을 검증에 배정했습니다. 학습은 16명 5,577개, 검증은 5명 1,775개이며 test는 사용하지 않았습니다. 중첩 창을 무작위로 나누면 같은 사람의 비슷한 데이터가 섞일 수 있어 사람 분할을 고정했습니다. 평균·표준편차·PCA도 학습 자료에서만 계산했습니다.

출처 및 근거
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/adapters/uci_reference.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/adapters/preprocessing_reference.py

## 슬라이드 4

[4분 경로 40초] 기준모델은 561개의 특징을 128개, 64개의 은닉 뉴런을 거쳐 6개 행동 점수로 바꾸는 MLP입니다. 첫 층은 561 곱하기 128에 편향 128개를 더하고, 모든 층을 합하면 80,582개 파라미터입니다. 시계열 버전은 128 곱하기 9를 펼칩니다. PCA 버전은 입력이 100차원으로 줄어 크기도 감소합니다. 마지막은 확률이 아닌 logits이며 손실 함수가 이를 처리합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/models/mlp.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/experiment_summary.csv

## 슬라이드 5

공통 trainer는 모델 구조를 만드는 함수와 분리했습니다. 따라서 다른 담당자가 만든 Keras 모델도 계약에 맞으면 연결할 수 있습니다. tf.data는 cache, 학습 shuffle, batch, prefetch를 사용합니다. 검증 데이터는 섞지 않습니다. seed 2026과 CPU 스레드를 고정했습니다. TFRecord는 별도 200표본 왕복 검사이며 일곱 실험은 배열 기반 tf.data로 학습했습니다. 재구성 모델 인터페이스의 연결 검사와 실제 팀 AE 학습은 구분합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/train.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/contracts.py
https://keras.io/api/callbacks/early_stopping/

## 슬라이드 6

[4분 경로 45초] 역전파는 손실의 기울기를 연쇄법칙으로 계산하고 옵티마이저는 이를 이용해 파라미터를 갱신합니다. 한 뉴런 예제에서 손실의 출력 기울기는 마이너스 3입니다. 출력의 가중치 기울기가 2이므로 곱해서 마이너스 6이 됩니다. 학습률 0.1이면 손실이 줄지만 0.5에서는 오히려 증가합니다. 별도로 17개 파라미터의 작은 tanh 신경망에서 직접 미분, 중심차분, 자동미분을 비교했습니다. 이 설명용 갱신은 SGD이며 실제 HAR 학습은 Adam입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/math_checks.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/math/gradient_checks.json
https://www.tensorflow.org/guide/autodiff

## 슬라이드 7

[4분 경로 55초] E02는 표준화한 특징 MLP이고, E03은 시계열, E07은 추가 표준화 생략, E08은 Dropout 0.3, E09와 E10은 학습률, E11은 PCA입니다. 최고 검증 Macro F1은 E02의 0.9138, Accuracy는 0.9161입니다. 정답은 1,775개 중 1,626개입니다. E08도 0.8940으로 기준선보다 낮았습니다. 이번 한 번의 검증 결과에서의 순위이며 최종 시험 성능이나 통계적 우월성은 아닙니다. 모델은 모두 최소 검증 손실 시점의 가중치를 사용했습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/experiment_summary.csv
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/EXPERIMENT_NOTES_KO.md

## 슬라이드 8

학습 손실은 계속 낮아지지만 검증 손실은 epoch 2에서 최소였습니다. 6개의 epoch 동안 그 최솟값을 갱신하지 못해 epoch 8에서 종료했습니다. 마지막 가중치가 아니라 epoch 2의 가중치를 복원해 저장합니다. 학습 곡선은 과적합 가능성을 보여 주지만 원인을 확정하지 않습니다. CSV의 epoch는 0부터 시작해 여기서는 1을 더해 표시했습니다. E02는 우연히 이 시점의 F1도 가장 높지만 선택 규칙 자체는 val_loss입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/train.py
실행 원본: runs/formal/E02_seed2026/history.csv
https://keras.io/api/callbacks/early_stopping/

## 슬라이드 9

표준화를 생략한 E07, Dropout을 추가한 E08, 학습률을 바꾼 E09와 E10 모두 이번에는 E02의 F1보다 낮았습니다. 여기서 퍼센트포인트는 F1 차이에 100을 곱한 값이며 상대 변화율이 아닙니다. E07은 검증 손실 0.2651로 E02보다 낮지만 F1은 낮습니다. 손실과 F1의 평가 대상이 다르기 때문에 생기는 차이입니다. 학습률과 조기 종료 정책의 상호작용도 있어 반복과 추가 실험이 필요합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/experiment_summary.csv
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

## 슬라이드 10

PCA 95%는 정확도 95%가 아니라 설명분산을 뜻합니다. 이 학습 자료에서는 100개의 성분이 선택되어 파라미터가 21,574개로 줄었습니다. 대신 검증 F1은 3.43퍼센트포인트 낮았습니다. 시계열 Flatten MLP 역시 입력과 파라미터 수가 달라 단순하게 시계열이 나쁘다고 말할 수 없습니다. 추론 시간이나 실제 메모리는 별도 측정이 필요하며 담당 ⑤의 검증으로 이어집니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/experiment_summary.csv
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/adapters/preprocessing_reference.py

## 슬라이드 11

오분류는 149개이고, 앉기와 서기 사이 61개, 걷기와 계단 오르기 사이 47개, 계단 오르기와 내려가기 사이 33개가 있습니다. 합계 141개로 오류의 약 94.6퍼센트입니다. 계단 오르기는 256개 중 212개를 맞혀 재현율 82.81퍼센트로 가장 낮습니다. 움직임의 유사성을 가능한 설명으로 제시할 수 있지만 행렬만으로 인과를 확정하지 않습니다. 이 분석은 검증 자료의 보조 관찰이며 최종 평가 담당 업무와 연결해야 합니다.

출처 및 근거
실행 원본: runs/formal/E02_seed2026/metrics.json
실행 원본: runs/formal/E02_seed2026/validation_predictions.csv

## 슬라이드 12

[4분 경로 45초] 마무리하면 기준모델뿐 아니라 공통 학습과 저장 형식까지 만들었습니다. 테스트 8개와 노트북 3개를 실행했고, 새 프로세스에서 모델 7개를 다시 불러 각 1,775개 예측이 같은지 확인했습니다. 같은 환경에서 최대 확률 차이는 0이었습니다. 이제 팀 전처리와 비교 모델을 연결하고 반복 seed, 공식 시험 평가, 시연을 검증해야 합니다. 저장 모델은 추론용 최선 가중치이며 옵티마이저 상태까지 정확한 재개 지점을 보장하는 체크포인트는 아닙니다. [4분 축약 발표는 여기서 종료하고 질문에 따라 보충 슬라이드를 사용합니다.]

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/VERIFICATION_KO.md
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/fresh_process_verification.json
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/TEAM_TASKS_KO.md

## 슬라이드 13

질문 시 참고하는 전체 수치 표입니다. 최선 epoch는 val_loss가 가장 낮은 epoch, 실행 epoch는 조기 종료까지 실제 실행한 길이입니다. E03은 10개 epoch, E10과 E11은 7개, 나머지는 8개를 수행했습니다. 시간은 장치와 최초 추적 비용의 영향을 받으므로 정밀한 속도 순위로 해석하지 않습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/experiment_summary.csv

## 슬라이드 14

중심차분은 L(theta+h)와 L(theta-h)의 차이를 2h로 나눈 근사입니다. 상대 오차는 두 기울기 차이의 절댓값을 max(1e-8, 두 기울기의 절댓값 합)으로 나눕니다. 단계 h에 따라 근사와 상쇄 오차가 달라질 수 있습니다. 이 작은 네트워크의 미분 일치를 검증한 것이며 모든 모델의 정당성을 증명하는 것은 아닙니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/reports/math/gradient_checks.json
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/math_checks.py

## 슬라이드 15

시계열 채널은 body_acc의 x,y,z, body_gyro의 x,y,z, total_acc의 x,y,z 순서입니다. 출력 logits는 softmax로 확률화할 수 있고 argmax로 클래스를 정합니다. 재구성 모델은 입력과 같은 shape를 출력하고 별도 task_type을 사용합니다. 다른 모델을 연결할 때 모델 생성 전에 seed를 설정하고 전처리 fit 범위도 동일하게 맞춰야 합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/contracts.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/src/adapters/uci_reference.py
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/HANDOFF_KO.md

## 슬라이드 16

환경은 Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.2.6, scikit-learn 1.7.2입니다. requirements.lock.txt에는 설치 버전 묶음이 있습니다. GitHub main에는 코드와 가벼운 근거 자료를 두고 v1 릴리스 ZIP에는 학습 모델을 포함했습니다. 공식 원시 자료는 저장소에 넣지 않고 내려받기 스크립트로 준비합니다. 재로딩은 compile=False로 하며 전처리를 다시 fit하지 않습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/requirements.lock.txt
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/RUN_GUIDE_KO.md
https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1

## 슬라이드 17

질문에는 관찰한 결과와 가능한 해석을 구분해 답합니다. 1. 검증 F1이 높다는 사실과 시험 성능은 다릅니다. 2. PCA는 분류 목적을 직접 최적화하지 않습니다. 3. Dropout 결과는 비율·모델·예산에 따라 달라질 수 있습니다. 4. 팀장으로서 구현된 인터페이스와 인계 자료는 준비했으나 실제 팀 모듈 통합과 발표 연습은 진행해야 합니다. 참고문헌은 보고서 마지막 쪽과 각 슬라이드 발표자 노트에 있습니다.

출처 및 근거
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/BONGHEON_WORK_SUMMARY.md
https://github.com/kara320090/smartphone-har/blob/704885df8776dcf594e65d19f0695991202b4eab/docs/EXPERIMENT_NOTES_KO.md
