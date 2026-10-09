# LSTM 발표 원고

본문 15장과 부록 5장입니다. 본문을 약 10분 설명하는 구성입니다.

## 슬라이드 1

본문 15장과 부록 5장입니다. 저는 LSTM 구현을 맡았습니다. 이번 발표는 데이터 연결과 LSTM 계산을 검증하고, 기본·소형·Dropout 조건을 반복 실행한 과정을 설명합니다. 공식 test는 평가하지 않았습니다. 약 10분 설명은 본문을 중심으로 하고 세부 질문은 부록과 보고서에서 답합니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/LSTM_EXECUTION_PLAN_KO.md

## 슬라이드 2

팀 역할은 데이터 정회서, MLP 한윤섭, CNN 최승빈, RNN 유재윤, LSTM 이봉헌입니다. 다른 팀원의 모델을 대신 구현하지 않았습니다. 세 조건은 결과 전에 정했고, 보조 조건이 좋아도 이를 자동으로 공동 비교 대표로 바꾸지 않습니다. 모든 모델에 같은 탐색 예산을 쓴 비교와 혼동하지 않기 위해서입니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/configs/L01.json
https://github.com/kara320090/smartphone-har/blob/main/configs/L02.json
https://github.com/kara320090/smartphone-har/blob/main/configs/L03.json

## 슬라이드 3

processed_data라는 이름만으로 이미 표준화되었다고 가정하지 않았습니다. 생성 코드와 통계를 확인하니 분할한 배열과 통계가 별도였습니다. 학습 배열에서 계산한 통계와 저장 파일이 원소별 일치했고, 공식 train의 모든 행과 9채널 값을 대조했습니다. UCI 구간은 이미 필터링과 윈도잉된 자료입니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/dataset_audit.json
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## 슬라이드 4

LSTM은 은닉 상태와 셀 상태를 갱신합니다. 입력 게이트와 후보 정보, 망각 게이트, 출력 게이트를 사용합니다. 이 식은 현대 Keras의 형태이며 1997년 논문의 기본 아이디어와 이후 망각 게이트 확장을 구분합니다. 셀 상태가 있다는 사실만으로 장기 의존성을 학습했다고 증명되지는 않습니다.

근거
https://keras.io/api/layers/recurrent_layers/lstm/
https://www.bioinf.jku.at/publications/older/2604.pdf
https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf

## 슬라이드 5

입력 차원은 9이고 은닉 크기는 64입니다. LSTM의 네 묶음 게이트마다 입력 kernel, recurrent kernel, bias가 있어 4H(D+H+1)로 계산합니다. 소형 조건은 LSTM뿐 아니라 다음 Dense 입력 크기도 달라집니다. 은닉 크기가 같은 RNN과도 총 파라미터는 다를 수 있으므로 규모 차이를 기록해야 합니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/src/models/lstm.py
https://github.com/kara320090/smartphone-har/blob/main/tests/test_lstm.py

## 슬라이드 6

단일 사람 분할을 고정했습니다. seed 반복은 이 분할에서 학습 변동을 보는 것이지 사람 모집단의 신뢰구간이 아닙니다. checkpoint는 F1이 아니라 val_loss로 선택합니다. 설정이나 실패 결과를 뒤늦게 덮어쓰지 않고 실행별 폴더를 보존합니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/LSTM_EXECUTION_PLAN_KO.md
https://github.com/kara320090/smartphone-har/blob/main/src/train.py

## 슬라이드 7

같은 seed라는 조건을 실제 초기 가중치 일치와 구분했습니다. L01 초기 배열을 저장하고 L03에 복사해 원소별로 비교했습니다. L02는 크기가 달라 같은 초기값이라고 주장하지 않습니다. 이후 Dropout의 확률적 경로 때문에 학습 궤적은 달라질 수 있으며 이는 비교하려는 조건입니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/src/train.py
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/initialization_audit.json

## 슬라이드 8

NumPy의 순전파와 BPTT를 자동미분 및 중앙차분과 비교했습니다. 자동미분 최대 차이는 2.776e-17, 중앙차분 최대 차이는 2.481e-10입니다. 미분을 일부러 틀리게 만든 대조는 실제 Keras 버그가 아닙니다. 테스트 개수를 모든 오류가 없다는 증거로 확대하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/math_verification.json
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/tests.xml

## 슬라이드 9

세 seed를 모두 표시했습니다. L01은 0.8813, L02는 0.8911, L03은 0.8550의 평균입니다. 조건별 순위는 이 설정과 분할 아래 관찰한 결과이며 보편적인 우월성이 아닙니다. 보조 조건 점수가 높더라도 공동 기준 L01을 자동 교체하지 않았습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/runs.csv
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/summary.csv

## 슬라이드 10

학습 손실이 내려가도 검증 손실이 개선되지 않을 수 있습니다. 손실은 정답 확률을 보고 F1은 분류 결과를 평가하므로 가장 높은 F1 epoch와 가장 낮은 손실 epoch가 다를 수 있습니다. 나중에 유리한 규칙으로 바꾸지 않았습니다. 나머지 두 seed의 곡선도 보고서에 모두 있습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/figures/learning_curves.png
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/runs.csv

## 슬라이드 11

혼동행렬은 대표 seed를 사전에 정해서 보여줍니다. 활동별 precision·recall·F1과 사람별 결과도 따로 보관했습니다. 혼동 원인을 신체 습관이나 센서 위치로 확정하지 않습니다. 원 신호와 추가 조건 분석 없이는 가능한 설명에 그칩니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/L01_seed2026_confusion.json
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/class_metrics.csv
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/errors.csv

## 슬라이드 12

같은 분할에서도 사람별 성능이 다릅니다. 전체 구간에서 클래스별 F1을 계산해 평균한 값과, 사람별 Macro F1을 먼저 계산한 뒤 평균한 값은 같지 않습니다. 사람마다 구간 수와 클래스 분포가 달라 support를 함께 제공합니다. seed 변동을 사람 모집단 신뢰구간으로 표현하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/subject_metrics.csv

## 슬라이드 13

대표 seed 2026의 세 조건을 표시하고 다른 여섯 실행도 원자료에 남겼습니다. 최초 tracing과 파일 로딩 및 Tensor 생성은 모델 시간에서 제외했습니다. 전처리 시간과 모델 시간의 median을 더해 전체 시스템 median이라고 부르지 않습니다. 학습 시간과 추론 시간의 측정 범위도 다릅니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/latency_summary.csv
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/latency_samples.csv
https://github.com/kara320090/smartphone-har/blob/main/src/lstm_analysis.py

## 슬라이드 14

단일 사람 분할이며 검증 자료는 epoch 선택에도 사용했습니다. 따라서 독립된 최종 평가와 구분합니다. RNN보다 높은지, CNN보다 빠른지는 팀원이 같은 기준으로 수행한 결과가 있어야 말할 수 있습니다. LSTM이 특정 시간 패턴을 왜 학습했는지도 별도 실험 없이는 추정입니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/lstm-report/REPORT_KO.md

## 슬라이드 15

핵심은 재현 가능한 결과와 설명 가능한 범위를 함께 넘기는 것입니다. L01 세 seed를 공동 기준으로 제공하고 L02·L03은 보조 분석으로 표시합니다. 코드와 저장물, 초기값 대조, 새 프로세스 재예측, 보고서와 발표 근거를 연결했습니다. 전체 논문은 다른 팀원의 실제 결과를 받은 뒤 취합합니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/lstm-report/LSTM_CONTRIBUTION_KO.md
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/summary.csv

## 슬라이드 16

미래에서 오는 셀 상태 기울기와 현재 은닉 상태를 통해 들어오는 기울기를 합칩니다. 망각 게이트의 preactivation 미분에는 sigmoid 미분이 필요합니다. 이를 의도적으로 빼면 자동미분과 차이가 나서 검사가 실패합니다. 전체 파라미터는 58개이고 세 간격으로 174회 검사했습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/src/lstm_math_checks.py
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/gradient_checks.csv

## 슬라이드 17

모든 정식 실행을 공개합니다. 최고 한 번의 결과만 사용하지 않았고 실패한 실행을 성공 실행으로 바꾸지 않았습니다. 작은 연결 검사는 정식 결과와 분리했습니다. 표의 값은 저장된 예측에서 다시 계산했습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/runs.csv

## 슬라이드 18

같은 seed와 같은 사람 분할의 차이를 대응시켜 제시합니다. 세 쌍의 값만으로 새로운 사용자 집단에 대한 확실한 우월성을 말하지 않습니다. L02는 파라미터 규모가 달라지고 L03은 같은 초기 가중치에서 외부 Dropout이 바뀌는 조건입니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/paired_differences.csv

## 슬라이드 19

파일명과 함수 이름만으로 입력 상태를 추정하지 않은 이유입니다. float64와 float32의 차이는 실제 배열 비교로 확인했지만 어느 쪽이 더 높은 정확도인지는 실험하지 않았습니다. 중복 표준화도 의도적 입력 대조이며 정식 실행에서 발생한 사고로 설명하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/dataset_audit.json

## 슬라이드 20

질문에는 실제 근거와 적용 범위로 답합니다. 테스트 수나 모델 이름 자체를 성능 보증으로 삼지 않습니다. 관련 HAR 연구에는 CNN과 LSTM 결합 모델도 있지만 이번 모델이 해당 논문을 재현했다고 표현하지 않습니다. 자세한 수식, 전체 수치, 재현 명령은 보고서와 질의응답 문서에 있습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/lstm-report/DEFENSE_QA_KO.md
https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/
