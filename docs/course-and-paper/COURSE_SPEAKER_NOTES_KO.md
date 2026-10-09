# 인공지능 공학 LSTM 수업 발표 원고

본문 15장 약 10~12분, 부록 5장은 질의응답용이다. 팀 30분 발표의 LSTM 8분 선택 구성은 COURSE_MAPPING_KO.md를 참고한다.

## 슬라이드 1

수업 발표용 개인 LSTM 파트입니다. 동일 프로젝트를 팀 공동 논문으로도 준비합니다. 본문 15장과 질문용 부록 5장이며 전체 본문 설명은 약 10~12분 구성입니다. 팀의 30분 발표에 맞춘 8분 선택 순서는 커리큘럼 연결 문서에 있습니다. 공식 test는 미평가입니다. 코드는 Codex 보조로 작성했고 실제 실행·수학 검산·저장 예측 대조를 수행한 근거를 설명합니다.

근거
인공지능 공학 강의자료 00.인공지능 공학 수업 OT.pdf, 2·10·14쪽. 개인 제공 자료, 공개 배포하지 않음.
https://github.com/kara320090/smartphone-har/blob/main/docs/LSTM_EXECUTION_PLAN_KO.md

## 슬라이드 2

OT는 수학 원리 설명, TensorFlow/Keras 구현, 데이터에서 평가까지의 팀 작업을 학습 목표로 제시합니다. 이 네 행은 새 채점표가 아니라 제 작업의 설명 순서입니다. 강의 후반 RNN/LSTM 주제는 OT 진도표와 공식 API에 연결했습니다.

근거
인공지능 공학 강의자료 00.인공지능 공학 수업 OT.pdf, 2·6·14쪽. 개인 제공 자료, 공개 배포하지 않음.
인공지능 공학 강의자료 02.신경망 기초 이론.pdf, 16·21·30쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 3

128을 채널 축으로 바꾸지 않습니다. UCI는 50Hz 신호를 처리하고 중첩 구간으로 제공하므로 완전한 원시 센서라고 부르지 않습니다. 전처리 담당 정회서의 결과를 재생성하지 않고 정확히 연결했습니다. 사람 분리는 새 사람에 대한 개발 평가 목적이며 validation은 epoch 선택에도 사용합니다.

근거
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/dataset_audit.json
인공지능 공학 강의자료 04_기계학습과_인식.pdf, 11·49·55쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 4

SimpleRNN의 tanh 예시와 LSTM의 상태 구조를 비교합니다. 실제 팀 RNN의 활성화와 유닛 수는 담당자의 구현을 확인해야 합니다. LSTM은 stateful=False, return_sequences=False이며 128번째 은닉 출력을 분류기에 사용합니다. 상태가 있다는 것만으로 긴 의존성을 학습했다는 증거가 되지는 않습니다.

근거
https://keras.io/api/layers/recurrent_layers/lstm/
인공지능 공학 강의자료 00.인공지능 공학 수업 OT.pdf, 6쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 5

네 preactivation을 나눈 뒤 입력·망각·출력 게이트에는 sigmoid를, 후보 정보에는 tanh를 적용합니다. 수식의 u_f는 망각 게이트 preactivation입니다. 후보는 양수·음수 정보를 표현할 수 있고 게이트는 0과 1 사이에서 통과 비율을 정합니다. Dense 분류기에는 ReLU를 사용합니다. 현대 Keras의 게이트 구조와 원 LSTM, 망각 게이트 확장을 구분합니다.

근거
https://keras.io/api/layers/recurrent_layers/lstm/
https://www.bioinf.jku.at/publications/older/2604.pdf
https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf
인공지능 공학 강의자료 05_신경망_기초.pdf, 29~32쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 6

원소별 곱의 미분과 덧셈의 역전파를 강의 개념에 연결합니다. c_(t-1)에서 c_t로 직접 전달되는 경로를 따라가면 미분은 f_t입니다. 게이트와 은닉 상태가 다시 연결된 전체 네트워크의 총 미분을 단순히 f_t 하나라고 쓰면 안 됩니다. 망각 게이트가 1에 가까우면 직접 경로의 감쇠를 줄일 수 있지만 실제 장기 기억 성공을 보장하지 않습니다.

근거
https://keras.io/api/layers/recurrent_layers/lstm/
인공지능 공학 강의자료 02.신경망 기초 이론.pdf, 30~33쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 7

이 식은 한 표본의 softmax 교차엔트로피입니다. 배치 평균이면 표본별 기울기에 배치 크기의 역수가 반영됩니다. 모델 최종 층에 softmax를 붙이지 않고 from_logits=True 손실을 사용합니다. 역전파는 기울기를 구하며 Adam은 그 기울기를 이용해 모멘트와 스케일을 반영하여 파라미터를 갱신합니다. Adam을 단순한 W-학습률*기울기와 같은 규칙이라고 설명하지 않습니다. 수식의 지수는 구현에서 안정적인 연산으로 처리합니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/src/models/lstm.py
https://github.com/kara320090/smartphone-har/blob/main/src/train.py
인공지능 공학 강의자료 05_신경망_기초.pdf, 23·31·41~44쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 8

dc와 dh는 각 상태에 대한 손실 기울기 누적값이며 미래에서 들어오는 항을 포함합니다. u_f 미분에는 sigmoid의 f(1-f)가 필요합니다. dc_(t-1) 식은 셀 상태의 직접 역전파 경로이고 은닉 상태 경로는 별도로 recurrent kernel을 통해 전달합니다. 전체 게이트·W·U·b 누적은 작은 NumPy 구현에서 확인할 수 있습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/src/lstm_math_checks.py
인공지능 공학 강의자료 02.신경망 기초 이론.pdf, 16·21·30쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 9

중앙차분은 한 파라미터를 조금 늘리거나 줄여 손실의 변화율을 구합니다. 수동 역전파, 자동미분, 세 차분 간격을 대조했습니다. 망각 게이트 sigmoid 미분을 의도적으로 빼서 검사가 실패하는지 확인했습니다. 이는 실제로 잘못 학습한 HAR 모델의 수리 기록이 아니라 검증 방법이 틀린 연산을 탐지한다는 대조입니다. 실제 학습은 Keras 자동미분을 사용합니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/math_verification.json
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/gradient_checks.csv
인공지능 공학 강의자료 02.신경망 기초 이론.pdf, 16쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 10

실제로 발견한 코드 간 연결 차이입니다. 입력 표준화 상태를 검사하고 원 행 번호를 보존했습니다. float64와 float32 표준화의 최대 입력 차이는 0.006851입니다. 이는 성능 차이 측정이 아닙니다. 두 번 표준화한 입력은 의도적 대조이며 실제 정식 실행 사고로 설명하지 않습니다. 기존 저장 모델도 새 연결 코드에서 전체 validation 재로딩 검증을 통과했습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/dataset_audit.json
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/legacy_bundle_verification.json
https://github.com/kara320090/smartphone-har/blob/main/src/adapters/processed_npz.py

## 슬라이드 11

D=9,H=64를 넣으면 LSTM 파라미터는 18,944개입니다. 구간별 상태는 독립적이고 Dropout은 Dense 뒤에서만 조건별로 변경합니다. 전체 코드 검사 39개와 저장 모델 9개의 새 프로세스 재예측을 확인했습니다. 확률 허용오차는 rtol 1e-5, atol 1e-6이고 예측 클래스는 일치했습니다. 테스트 수 자체를 모든 버그가 없다는 보증으로 설명하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/src/models/lstm.py
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/delivery_verification.json

## 슬라이드 12

하이퍼파라미터는 학습으로 찾는 가중치와 구분합니다. 은닉 크기와 외부 Dropout은 결과 전에 정했고 최대 40epoch, Adam .001, batch64를 고정했습니다. 같은 seed만 믿지 않고 초기 가중치를 실제로 맞췄습니다. L02는 규모가 달라 동일 초기 배열이라고 주장하지 않습니다. validation 선택 기준은 loss이며 F1 최고 epoch로 바꾸지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/configs/L01.json
https://github.com/kara320090/smartphone-har/blob/main/configs/L02.json
https://github.com/kara320090/smartphone-har/blob/main/configs/L03.json
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/initialization_audit.json
인공지능 공학 강의자료 05_신경망_기초.pdf, 41~44쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 13

F1은 클래스별 precision과 recall을 결합하고 Macro F1은 여섯 클래스 F1의 평균입니다. L01 평균은 0.8813, L02는 0.8911, L03은 0.8550입니다. L02가 이 설정에서 높았다는 관찰을 보편적 최적 구조로 확대하지 않습니다. L01을 공동 기준으로 유지합니다. 3seed SD는 사람 모집단의 신뢰구간이 아닙니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/runs.csv
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/summary.csv
인공지능 공학 강의자료 04_기계학습과_인식.pdf, 52~53쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 14

검증 손실은 학습 손실과 함께 내려갈 필요가 없습니다. 과적합 가능성은 곡선을 보며 설명하지만 실제 사용자 일반화의 원인을 확정하지 않습니다. cross-entropy와 F1은 서로 다른 지표여서 가장 낮은 loss의 epoch와 가장 높은 F1 epoch는 다를 수 있습니다. 최선 val_loss를 기준으로 선택했고 독립 최종 평가가 남아 있습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/runs/lstm/L01_seed2026/history.csv
인공지능 공학 강의자료 04_기계학습과_인식.pdf, 49·55쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 15

RUN_DEMO 문서의 명령으로 저장한 L01 seed2026 모델을 불러오고 validation 다섯 구간을 분류합니다. 준비된 데이터의 오프라인 시연이고 센서 수집 앱이나 공식 test가 아닙니다. 예측이 틀린 경우도 그대로 설명합니다. 수업에서는 제가 이해한 원리와 검증 과정을, 공동 논문에서는 팀의 네 모델 비교를 중심으로 설명합니다. 팀 결과와 공식 test는 아직 취합·평가되지 않았습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/course-and-paper/RUN_DEMO_KO.md
https://github.com/kara320090/smartphone-har/blob/main/docs/course-and-paper/TEAM_INTEGRATION_KO.md
인공지능 공학 강의자료 00.인공지능 공학 수업 OT.pdf, 10·14쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 16

모든 정식 실행의 결과입니다. smoke 검사는 포함하지 않았습니다. 예측 CSV에서 지표를 재계산한 수치이며 최고 seed만 선택하지 않았습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/runs.csv

## 슬라이드 17

사람별 표본 수와 클래스 분포가 달라 support도 보관합니다. 전체 구간의 Macro F1과 사람별 Macro F1의 평균은 동일한 값이 아닙니다. 5명을 새로운 사용자 모집단 전체로 일반화하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/subject_metrics.csv

## 슬라이드 18

강의 기계학습과 인식 50쪽의 교재 n_ij는 행이 예측, 열이 실제입니다. 53쪽은 sklearn과 교재 방향이 반대라고 명시합니다. 이 프로젝트는 sklearn 기준 행=정답, 열=예측을 사용했습니다. 축 방향을 읽지 않으면 앉기를 서기로 오분류한 것과 반대 방향을 혼동합니다. 148개는 이 대표 seed의 관찰이고 사용자 습관이나 센서 위치로 원인을 확정하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/L01_seed2026_confusion.json
인공지능 공학 강의자료 04_기계학습과_인식.pdf, 50·53쪽. 개인 제공 자료, 공개 배포하지 않음.

## 슬라이드 19

같은 CPU에서 suite 종료 후 측정했습니다. 입력 Tensor 생성, tracing, 파일 로딩은 모델 호출 시간에 포함하지 않았습니다. median과 p95를 구분하며 개인 PC 간 측정값으로 팀 모델 속도 순위를 정하지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/reports/lstm/latency_summary.csv
https://github.com/kara320090/smartphone-har/blob/main/src/lstm_analysis.py

## 슬라이드 20

상세 질의응답은 기존 보고서 자료에 있습니다. AI 보조 사용은 밝히고 실제 검증한 계산과 결과를 설명합니다. 같은 프로젝트의 수업 발표는 정식 논문 출판과 구분하며 학회 규정을 따릅니다. 연구의 독창성은 모델 이름이나 개수만으로 보장되지 않습니다.

근거
https://github.com/kara320090/smartphone-har/blob/main/docs/lstm-report/DEFENSE_QA_KO.md
https://github.com/kara320090/smartphone-har/blob/main/docs/course-and-paper/TEAM_INTEGRATION_KO.md
https://www.aps.org/about/governance/policies-procedures/ethics-standards
