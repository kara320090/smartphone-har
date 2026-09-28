# 이봉헌 담당 업무 정리

담당은 **② 기준모델 구현·공통 학습·조건 비교와 팀장**이다. 이 문서는 구현한 코드, 실제 실행 결과, 확인한 근거와 남은 팀 연결 업무를 한곳에 정리한다.

## 후속 심화 검증

2026년 9월 29일 입력 계약 결함을 재현·수정하고 신규 16개 포함 총 24개 회귀 검사를 통과했다. 기존 7조건을 3seed에서 다시 학습하고 초기 가중치를 일치시킨 Dropout 대조 3회를 추가했다. 후속 학습 24개 모델의 재로딩을 확인했고, seed 2026의 기존 예측도 재학습 후 일치했다.

[심화 보고서·PPT·질의응답](process-report/README.md), [실제 과정 기록](process-report/PROCESS_LEARNING_LOG_KO.md), [반복 결과](../reports/process_audit/summary.json)를 함께 본다. 아래 표와 단일 seed 점수는 최초 v1의 기록이다.

## 최초 완료한 업무

| 업무 | 구현과 산출물 | 완료 근거 |
|---|---|---|
| 특징 MLP | 561 → 128 ReLU → 64 ReLU → 6 logits | E02 실제 학습, 파라미터 80,582개 |
| 시계열 MLP | 128×9 → Flatten → 128 → 64 → 6 logits | E03 실제 학습, 파라미터 156,230개 |
| 공통 학습 | Adam, 조기 종료, 최선 가중치 복원, 설정·로그·모델 저장 | MLP 및 외부 분류·재구성 모델 연결 검사 |
| 역전파 검증 | 손계산, NumPy, 중심차분, TensorFlow 자동미분 | 17개 파라미터 × h 3종, 총 51개 비교 |
| 조건 비교 | 표준화, Dropout, 학습률, PCA | E07–E11 설정과 실제 검증 결과 |
| 데이터 파이프라인 | tf.data, TFRecord 저장·복원 | 학습 200개 표본의 값·shape·정답·ID 완전 일치 |
| 저장물 인계 | 모델·전처리·메타데이터·분할·표본별 예측 | 새 프로세스에서 7개 모델의 검증 예측 재현 |
| 설명과 재현 | 노트북 3개, 버전 고정, 실행·인계·발표 문서 | 노트북 전체 실행, 테스트 8개 통과 |

담당 업무를 독립적으로 진행하기 위한 UCI 참조 어댑터를 포함했다. ①의 분석·군집 작업이나 ③의 실제 CNN/RNN/LSTM/AE 구현을 완료한 것으로 표시하지 않는다.

## 실제 실험 결과

학습은 16명 5,577개, 검증은 사람 1·6·14·21·23의 1,775개다. 전처리는 학습 자료로만 계산한다. 모든 값은 **seed 2026의 검증 결과**이며 공식 시험 점수가 아니다.

| 실험 | Accuracy | Macro F1 | 파라미터 | 최선 / 실행 epoch |
|---|---:|---:|---:|---:|
| E02 | 0.9161 | 0.9138 | 80,582 | 2 / 8 |
| E03 | 0.8783 | 0.8765 | 156,230 | 4 / 10 |
| E07 | 0.8873 | 0.8864 | 80,582 | 2 / 8 |
| E08 | 0.8952 | 0.8940 | 80,582 | 2 / 8 |
| E09 | 0.8772 | 0.8749 | 80,582 | 2 / 8 |
| E10 | 0.8913 | 0.8899 | 80,582 | 1 / 7 |
| E11 | 0.8811 | 0.8795 | 21,574 | 1 / 7 |

이번 7개 중 E02가 검증 macro F1 0.9138로 가장 높았다. E03은 입력 표현과 모델 크기가, E11은 PCA 차원과 첫 층 크기가 함께 달라진다. 단일 seed 결과를 통계적 우월성으로 해석하거나 전체 팀 모델 선택 결과로 확정하지 않는다.

[상세 결과와 학습 곡선](../reports/RESULTS_KO.md) · [조건별 가설과 해석](EXPERIMENT_NOTES_KO.md)

## 담당 파일

| 위치 | 내용 |
|---|---|
| [src/models/mlp.py](../src/models/mlp.py) | MLP 두 종류 |
| [src/train.py](../src/train.py) | 공통 학습·저장·조기 종료 |
| [configs](../configs/) | 실험 7개 설정 |
| [src/contracts.py](../src/contracts.py) · [참조 어댑터](../src/adapters/) | 팀 데이터 연결과 학습용 입력 |
| [src/math_checks.py](../src/math_checks.py) | 미분·역전파 수치 검증 |
| [src/tfrecord_check.py](../src/tfrecord_check.py) | Data API 연계 실습 |
| [src/verify_bundle.py](../src/verify_bundle.py) | 저장 전처리와 모델의 예측 재현 |
| [notebooks](../notebooks/) | 수학, 학습 흐름, 조건 비교 설명 |
| [evidence_index.csv](../evidence_index.csv) | 강의 개념과 근거 파일 대응 |

## 실행과 공유

- 처음 실행할 때: [시작 안내](../START_HERE_KO.md) → [상세 실행](RUN_GUIDE_KO.md)
- 이미 학습된 모델 사용: [GitHub v1 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1)의 전체 ZIP
- 팀원에게 전달할 입력·출력·저장 형식: [인계 문서](HANDOFF_KO.md)
- 발표 준비: [약 4분 원고와 예상 질문](PRESENTATION_KO.md)

각 실행 폴더에는 model.keras, preprocess.npz, metadata.json, config.json, split.json, history.csv, validation_predictions.csv, metrics.json과 검사 기록을 보관한다. 가중치는 최저 검증 손실 시점이며, optimizer 상태와 함께 정확히 학습을 재개하는 체크포인트는 지원하지 않는다.

## 아직 팀과 함께 확인할 업무

- [ ] ①의 공식 모듈과 sample_id·분할·전처리 수치 일치 확인
- [ ] ③의 비교 모델을 공통 trainer에 연결
- [ ] ④의 전체 모델 반복 seed와 공식 시험 평가 연결
- [ ] ⑤의 시연·입력 오류 처리·시간 측정에 저장물 인계
- [ ] 다른 팀원 환경에서 실제 실행 확인
- [ ] 각자 근거를 설명하고 발표·질의응답 연습

이 항목은 실제 팀 협업과 개인 설명이 필요하다. [팀장 작업표](TEAM_TASKS_KO.md)로 확인한다.

## 데이터와 근거

데이터는 [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), DOI 10.24432/C54S4K, CC BY 4.0이다. 구현 범위와 실험 설정은 제공된 10주 상세계획서에 근거한다. [검증 기록 요약](../reports/VERIFICATION_KO.md)에서 실제 확인 범위를 볼 수 있다.
