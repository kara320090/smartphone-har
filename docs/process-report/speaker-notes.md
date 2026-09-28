# 이봉헌 구현 과정 심화 발표 원고

본문 13장과 보충 7장입니다. 전체 설명은 약 10–12분을 기준으로 하며 말하는 속도에 따라 달라집니다.

4분 축약 경로: **1 → 3 → 4 → 6 → 7 → 9 → 13**. 목표 시간은 15+35+30+40+35+30+55=240초이며 각 노트의 축약 문단을 사용합니다.

## 슬라이드 1

[본문 13장, 보충 7장 / 약 10–12분 설명용]
[4분 축약 경로: 1 → 3 → 4 → 6 → 7 → 9 → 13]
[축약 15초] 기준모델 구현을 맡은 이봉헌입니다. 이번에는 성능표뿐 아니라 실제 입력 오류를 어떻게 찾아 수정했고, 비교 실험의 통제 조건을 어떻게 검증했는지 설명하겠습니다. 기존 보고서 이후 실제로 수행한 후속 작업입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/docs/PROCESS_AUDIT_PLAN_KO.md

## 슬라이드 2

잘못된 인덱스 처리와 원본 검증 자료의 정렬 검사는 실제 기존 코드의 결함이었습니다. 같은 seed의 가중치 차이는 구조를 직접 생성해 확인한 통제의 한계입니다. 배치 평균을 빠뜨린 미분이나 naive softmax는 검사가 오류를 잡는지 확인하려고 일부러 구성했습니다. 이런 대조를 개발 중 우연히 겪은 사고로 이야기하지 않습니다. seed별 성능 차이 역시 버그와 다릅니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/contracts_before.json
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/math_controls.json

## 슬라이드 3

[축약 35초] 기존 코드는 분할 번호를 바로 int64로 바꿨습니다. 소수는 잘리고, 참거짓은 1과 0이 되고, 음수는 뒤쪽 행을 고릅니다. 합성 18행 입력에서 이 동작을 실제로 확인했습니다. 수정 후에는 먼저 정수 벡터와 범위, 중복을 확인한 뒤 변환합니다. 공식 UCI 로더의 정상 인덱스는 원래 유효했으므로 이전 성능이 틀렸다고 단정하지 않고 다시 학습해 비교했습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/src/contracts.py
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/contracts_before.json
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/contracts_after.json

## 슬라이드 4

[축약 30초] 학습용 전처리 결과와 저장물 검증용 원본을 따로 가지고 있는데, 기존에는 길이만 같으면 통과했습니다. 원본 순서를 뒤집어도 통과했고 최대 값 차이가 11.6729였습니다. 이제 저장된 전처리를 다시 적용해 전체 배열이 맞는지 검사합니다. 이 검사는 자기 일관성을 확인하므로 원본과 ID가 함께 잘못된 상황까지 모두 증명하는 것은 아닙니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/contracts_before.json
https://github.com/kara320090/smartphone-har/blob/main/src/contracts.py

## 슬라이드 5

15실패는 버그 15개라는 뜻이 아닙니다. 같은 원인을 여러 경계 입력에서 점검한 결과입니다. 잘못된 분할에서 Preprocessor.fit이 호출되면 검사 자체가 실패하도록 해 차단 시점도 보았습니다. 정상 입력을 유지하는 양성 대조로, 검증 입력에 10,000을 더해도 학습 평균·표준편차와 학습 입력이 같은지 확인했습니다. 기존 모델 저장과 외부 모델 연결 검사까지 포함해 총 24개가 통과했습니다. 라이브러리 deprecation 경고 26개는 남았습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/tests/test_contract_regressions.py
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/regressions_before.xml
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/tests_after.xml

## 슬라이드 6

[축약 40초] 처음에는 같은 seed를 사용해 조건을 대응시켰습니다. 그런데 직접 가중치를 비교해 보니 Dropout 모델의 둘째 Dense와 출력층이 달랐습니다. 구조 변경에 따라 난수 사용 순서가 달라질 수 있기 때문입니다. 같은 seed 설정 비교로는 유효하지만 초기값까지 같다고 말하면 안 됩니다. 그래서 이 발견을 기록한 뒤 E02의 모든 Dense 초기값을 복사하는 E08W 대조를 세 seed에서 추가했습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/initialization_control.json
https://github.com/kara320090/smartphone-har/blob/main/docs/PROCESS_AUDIT_AMENDMENT_KO.md

## 슬라이드 7

[축약 35초] E08W는 Dropout 0.3 구조를 유지하고 E02의 초기 Dense 가중치와 편향을 모두 복사했습니다. 배열 완전 일치와 해시를 확인한 뒤 학습했습니다. 그래프는 세 seed의 모든 결과입니다. 초기값이라는 혼동 요인 하나를 제거했지만 Dropout 마스크와 학습 경로, 조기 종료 시점은 다릅니다. 이 결과도 현재 구조와 비율의 관찰이지 Dropout의 보편적 효과를 증명하지 않습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/analysis/process_audit/run_matched_dropout.py
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/condition_summary.csv

## 슬라이드 8

기존 7조건은 실행 전에 정한 seed 세 개 모두에서 학습했습니다. 초기값 대조는 별도 계획으로 3회를 추가해 총 24회입니다. 표준편차는 ddof=1입니다. 같은 사람 분할에서의 초기화·셔플 변동을 보여 주며 다른 사람을 새로 추출한 변동은 아닙니다. E08W는 원래 7조건의 사전 계획에 포함됐던 것처럼 숨기지 않고 별도 대조로 보고합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/docs/PROCESS_AUDIT_PLAN_KO.md
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/condition_summary.csv

## 슬라이드 9

[축약 30초] seed 2026에서는 E02가 가장 높았지만 2027과 2028의 최고 조건은 달랐습니다. 따라서 한 번의 결과로 기본 설정이 항상 우수하다고 결론 내릴 수 없습니다. 이번 보강은 평균뿐 아니라 표준편차와 seed별 차이를 모두 보고하고, 현재 분할의 관찰이라는 범위를 명시했습니다. 평균 차이가 작다는 이유만으로 세 번의 결과에 유의성 검정을 덧붙이지 않았습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/repeat_results.csv
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/paired_deltas.csv

## 슬라이드 10

UCI의 창은 2.56초이며 50퍼센트 중첩됩니다. 같은 사람의 창을 독립적으로 나눠 학습과 검증에 섞으면 새로운 사람에게 일반화하는 상황을 제대로 평가하기 어렵습니다. 사람 기준 분할을 유지했고 표준화와 PCA도 학습 자료로만 fit했습니다. 1,775개 창을 독립 표본으로 취급하는 신뢰구간은 제시하지 않았습니다. 또 같은 검증 자료로 조기 종료와 조건 선택을 했으므로 최종 평가는 남겨 둔 test에서 별도로 해야 합니다.

출처 및 근거
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
https://github.com/kara320090/smartphone-har/blob/main/src/adapters/uci_reference.py
https://github.com/kara320090/smartphone-har/blob/main/docs/PROCESS_AUDIT_PLAN_KO.md

## 슬라이드 11

실제 E02처럼 ReLU와 6클래스 교차엔트로피 평균을 사용하는 작은 네트워크를 별도로 만들었습니다. 46개 파라미터를 h 세 크기에서 검사했고 자동미분과도 일치했습니다. 그런데 ReLU의 0에서는 중심차분이 0.5이고 TensorFlow는 0을 선택합니다. 미분계수가 유일하지 않은 점이므로 이를 그대로 버그로 해석하면 안 됩니다. 반면 평균 손실에서 배치로 나누는 항을 빼면 8배의 잘못된 기울기가 되어 검사가 잡았습니다. 이는 의도적 대조입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/analysis/process_audit/math_controls.py
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/math_controls.json

## 슬라이드 12

logits에 큰 값이 들어가면 naive softmax는 exp에서 overflow가 발생합니다. 최대 logits를 빼면 softmax 확률은 유한해집니다. 하지만 매우 작은 정답 확률이 0이 되면 마이너스 로그 0은 여전히 무한대입니다. 그래서 학습 손실은 logits 기반 logsumexp 형태로 계산합니다. 정답을 매우 작은 확률의 클래스로 둔 예에서도 유한한 손실을 얻었고 TensorFlow 결과와 일치했습니다. 실제 HAR에서 이 사고가 났다고 주장하는 사례가 아니라 구현 선택을 검증하는 대조입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/math_controls.json
https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits

## 슬라이드 13

[축약 55초] 정리하면 실제 코드 경계의 오류를 재현하고 차단했으며, 정상 UCI 입력으로 기존 7조건을 다시 학습했을 때 이전 예측과 완전히 일치했습니다. 같은 seed가 같은 초기값을 뜻하지 않는다는 반론도 직접 확인하고 가중치를 맞춘 대조를 추가했습니다. 결론도 한 번의 최고 점수에서 평균과 변동을 함께 설명하는 형태로 수정했습니다. 다만 같은 검증 사람 5명과 세 seed의 범위를 넘는 일반화는 남아 있습니다. 팀 모듈 통합, 다른 사람 분할, 최종 test, 시연 검증을 다음 단계로 제시합니다. [축약 경로의 목표 시간 합계 240초]

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/summary.json
https://github.com/kara320090/smartphone-har/blob/main/docs/PROCESS_AUDIT_PLAN_KO.md

## 슬라이드 14

행 배치 표기에서 X는 B 곱하기 561, 첫 가중치는 561 곱하기 128입니다. 출력 오차 D3의 shape는 B 곱하기 6이고 H2 전치는 64 곱하기 B이므로 출력 가중치 기울기는 64 곱하기 6입니다. ReLU의 마스크는 preactivation이 양수인지를 사용합니다. 손실이 평균이면 1/B를 출력 오차에서 한 번 반영하고, 편향은 배치 축으로 합합니다. 실제 모델은 Keras 자동미분으로 학습했고 수동 미분은 별도 작은 네트워크에서 검증했다는 범위를 명확히 합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/src/models/mlp.py
https://github.com/kara320090/smartphone-har/blob/main/src/train.py
https://github.com/kara320090/smartphone-har/blob/main/analysis/process_audit/math_controls.py

## 슬라이드 15

같은 검증 사람에서도 성능이 다르고 seed 변동도 보입니다. 모든 창을 합쳐 클래스별 F1을 계산한 Macro F1과, 각 사람의 Macro F1을 먼저 구해 평균한 값은 일반적으로 다릅니다. 사람별 창 수와 클래스 분포도 확인해야 합니다. 현재 다섯 사람만으로 새로운 사람 모집단의 신뢰구간을 제시하지 않았습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/e02_by_subject.csv

## 슬라이드 16

seed 2026의 E02는 epoch 2의 검증 손실이 최소이고 6개 epoch 동안 더 좋아지지 않아 8에서 종료했습니다. E07은 손실이 더 낮아도 F1은 낮습니다. 지표의 평가 방식이 다르기 때문입니다. 모든 실행의 선택 규칙은 처음부터 val_loss였고 결과를 보고 F1로 고른 epoch와 섞지 않았습니다. F1 기준 선택을 비교하려면 별도 정책으로 실험해야 합니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/summary.json
https://keras.io/api/callbacks/early_stopping/

## 슬라이드 17

모델을 다시 학습해 같은 결과가 나오는지와, 저장한 모델을 불러 같은 예측이 나오는지는 서로 다른 검사입니다. 입력 계약을 바꾼 이후에도 seed 2026의 7조건은 기존 예측과 일치했습니다. 새 실행 24개도 원본 검증 입력부터 저장 전처리를 거쳐 다시 예측했습니다. 저장 가중치는 최선 epoch이지만 옵티마이저는 마지막 상태일 수 있어 정확한 학습 재개 체크포인트라고 부르지 않습니다. 다른 하드웨어의 비트 단위 동일성도 검증 범위 밖입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/summary.json
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/reload_repeats.json
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/reload_matched.json

## 슬라이드 18

무슨 질문에도 반박할 수 없도록 만드는 것은 현실적인 목표가 아닙니다. 대신 각 주장에 정확한 근거와 적용 범위를 연결합니다. 실제로 검사한 입력을 제시하고, 원본 의미 확인이나 전체 장치 호환성과 같은 다른 문제는 따로 남깁니다. 테스트 개수만 늘리는 대신 실패 사례와 정상 사례를 함께 확인한 이유도 이 때문입니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/tests/test_contract_regressions.py
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/tests_after.xml

## 슬라이드 19

PCA는 학습 특징의 분산을 보존하는 기준이지 정답 분류율을 보장하지 않습니다. 세 seed의 SD는 학습 변동을 보여 주며 모집단 신뢰구간이 아닙니다. 최고 평균도 검증을 반복해서 살핀 선택의 결과이므로 test 성능과 구분합니다. 다른 모델 담당자의 구현과 최종 평가, 실제 시연을 제가 완료한 것으로 표시하지 않습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/docs/HANDOFF_KO.md
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/summary.json

## 슬라이드 20

이 슬라이드는 준비된 질문을 이용해 사고 과정을 보여 주기 위한 것입니다. 일부러 틀린 결과나 겪지 않은 사건을 말하지 않습니다. 질문을 던진 다음 최소 재현 입력, 실제 수치, 수정 또는 선택의 이유를 설명합니다. 답하지 못하는 질문은 현재 근거가 없는 범위를 밝히고 어떤 후속 검사로 답할지 제안합니다. 확장 질의응답 문서에는 증거 파일과 피해야 할 과장 표현을 함께 정리했습니다.

출처 및 근거
https://github.com/kara320090/smartphone-har/blob/main/docs/process-report/DEFENSE_QA_KO.md
https://github.com/kara320090/smartphone-har/blob/main/reports/process_audit/math_controls.json
