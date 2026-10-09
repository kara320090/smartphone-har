# 이봉헌 LSTM 실행 계획

2026년 10월 9일 사용자 승인 계획. 구현·학습 전에 저장한다.

## 목표와 범위

이봉헌 담당 LSTM의 구현·학습·평가·검증과 팀 논문 원고, 개인 보고서 MD·Word, 발표 PPT를 완성한다. MLP·CNN·RNN은 다른 팀원의 영역이다. 기존 MLP 작업과 결과는 보존한다. 공식 test 성능 평가는 이번 범위에 포함하지 않는다.

입력은 `C:/Users/soma/Desktop/processed_data.npz`, 통계는 저장소의 `data/preprocess.npz`다. 데이터 SHA256은 `00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459`, 통계 SHA256은 `42b1b895d420a21ed7cce3b3636a4cbed96350953b27057c0670db9062d79fb2`다. fit 16명 5,577구간, validation 5명 1,775구간이며 검증 사람은 1·6·14·21·23이다. 배열은 분할되었으나 표준화는 적용 전이다. 저장 통계가 fit 배열과 일치함을 확인했으며 이를 float32 연산으로 한 번 적용한다.

## 구현 규약

- `--processed-data`, `--preprocess-stats` 입력을 기존 학습 CLI에 추가한다. 원본 기반 MLP 실행은 유지한다.
- fit·val 배열만 접근한다. test 배열·정답은 읽거나 평가하지 않는다.
- 0 기반 sample_id를 보존하고 데이터·통계 해시, 사람·채널·라벨·표준화 상태를 기록한다.
- 모델은 Input(128,9), LSTM, Dense(32,relu), Dropout, Dense(6 logits)다. 독립 구간, stateful=False, 마지막 시점 출력, tanh·sigmoid, LSTM 내부 dropout=0, recurrent_dropout=0이다.
- logits와 SparseCategoricalCrossentropy(from_logits=True)를 사용한다. 모델 선택에 lstm을 추가하되 mlp는 유지한다.
- 새 전처리 저장 형식을 구분하며 기존 reference_npz_v1 저장물도 검증 가능하게 유지한다.

## 정식 실험

| 조건 | units | 외부 Dropout | 질문 |
|---|---:|---:|---|
| L01 | 64 | 0.0 | 공동 비교 기준과 학습 변동 |
| L02 | 32 | 0.0 | 축소 시 성능·규모 차이 |
| L03 | 64 | 0.3 | 초기값을 일치시킨 Dropout 차이 |

각 조건 seed 2026·2027·2028, 총 9회. Adam lr=0.001, batch=64, epochs<=40, val_loss patience=6, 최선 가중치 복원, CPU threads=4. 실행마다 별도 프로세스이며 순차 실행한다. 작은 입력 연결·시간 점검은 9회에 포함하지 않는다. L03은 같은 seed L01의 **학습 전** 가중치를 복사·검증한다. L02는 크기가 달라 같은 초기값이라고 표현하지 않는다. 공동 비교에는 L01 세 seed를 제공하고 L02·L03은 별도 보조 분석으로 둔다.

## 검증과 해석

- 입력·라벨·ID·사람·통계·정렬 검사, 팀 표준화 수치 대조, 잘못된 입력 탐지, 기존 MLP와 새 LSTM 테스트.
- 작은 LSTM NumPy 순전파·BPTT와 TensorFlow 자동미분·중앙차분 비교. 의도적 미분 오류 대조를 실제 결함과 구분한다.
- 실제 문제는 재현 입력·관찰·원인·수정·재검증을 기록한다. 없었던 실패를 만들지 않는다.
- 새 프로세스에서 각 모델의 validation 1,775개 전체 재예측. 클래스 일치, 확률 rtol=1e-5·atol=1e-6.
- Accuracy, 6클래스 Macro F1, precision·recall·F1, 사람별 support·지표, 혼동행렬, 학습 곡선, epoch·파라미터·시간, 오류 ID를 보관한다.
- seed 평균·표본 SD 및 같은 seed의 대응 차이를 보고한다. SD는 사람 모집단 신뢰구간이 아니다. 목표 정확도는 설정하지 않는다.
- CPU batch 1 추론은 준비된 입력을 사용해 50회 워밍업 뒤 500회 측정한다. 출력 반환까지 포함하고 전처리는 별도 측정한다. PC 값을 스마트폰 실시간 성능으로 표현하지 않는다.

## 산출물과 완료 기준

코드·설정·실행 안내·테스트, 9개 모델과 실행 근거, 상세 MD·Word 보고서(약 20쪽), PPT 본문 15장과 부록 5장, 발표 원고·질의응답, 팀 논문용 LSTM 원고·구조표·결과표, GitHub 소스·문서·Release를 제공한다. 원고의 수치는 동일 결과 파일에서 생성한다. Word 모든 페이지와 PPT 모든 슬라이드를 렌더링·시각 검토한다.

완료 기준은 9회 실행, 지표 재계산, 전체 검증 자료 재로딩, 코드 테스트, 문서 수치 대조, GitHub 게시다. 팀원 모델 비교와 공식 test는 후속 공동 작업이다.

## 변경 기록

| 시점 | 변경 | 이유 |
|---|---|---|
| 구현 시작 전 | 승인 계획 저장, 9회 조건 고정 | 결과를 본 뒤 조건을 바꾸지 않기 위함 |
| 소량 검사 후 | 39개 코드 검사, 3개 소량 조건과 수학 검산 통과; 95422e0에서 학습 코드 고정 | 정식 실행 전에 연결 확인 |
| 9회 완료 후 | 예측에서 지표 재계산, 모델·전처리 각각 50회 워밍업·500회 시간 측정 | 학습 중 CPU 작업과 추론 시간 측정을 분리 |
| 결과 분석 후 | 오류 ID에 9채널 평균·SD·RMS와 원본 행 대조 재현 명령 추가 | 관찰 가능한 오류 근거와 재현성 보강; 학습 조건 변경 없음 |
| 문서 작성 후 | 약 20쪽 목표를 실제 22쪽으로 조정 | 전체 9회 결과·수학·오류 표본·한계를 읽기 좋은 배치로 보존 |
| 문서 검증 | 기본 Word 렌더러 부재로 Word PDF·Poppler 사용, PPT native 표·차트 검사 및 전 페이지 시각 검토 | 현재 Windows에서 실제 표시를 확인; 검증 방법 명시 |

## 진행 체크

- [x] 데이터 연결·LSTM·CLI 구현
- [x] 작은 입력·수학·회귀 검증
- [x] 정식 9회 학습
- [x] 전체 재로딩·지표 재계산·시간 측정
- [x] 보고서·PPT·원고·질의응답
- [ ] GitHub·Release 게시

출처: [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/).
