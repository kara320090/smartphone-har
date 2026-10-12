# 제공 main과 전처리 데이터 확인 결과

확인일 2026년 10월 12일

제공된 `processed_data.npz`는 main의 LSTM 실험 설정에서 고정한 데이터 버전과 정확히 같다. 데이터 SHA256은 `00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459`이며 `data/preprocess.npz`의 SHA256도 설정과 일치한다.

| 확인 항목 | 실제 상태 |
|---|---|
| fit 시계열 | float32 5,577 × 128 × 9 |
| validation 시계열 | float32 1,775 × 128 × 9 |
| validation 사람 | 1, 6, 14, 21, 23 |
| fit 사람 | 3, 5, 7, 8, 11, 15, 16, 17, 19, 22, 25, 26, 27, 28, 29, 30 |
| 라벨 | int64 0–5, 두 개발 집합 모두 여섯 클래스 |
| 사람·sample_id 중복 | fit와 validation 사이 없음 |
| 비유한 값 | fit·validation 시계열에 NaN·Inf 없음 |
| 저장 통계 | full fit 시계열에서 계산한 평균·표준편차와 정확히 일치 |
| 캐시 표준화 | 아직 적용 안 됨, 저장 통계로 한 번 적용 필요 |
| 원본 sample_id | 공식 train의 0 시작 행 번호 |
| 원본 모델 registry | MLP·LSTM만 등록 |
| 기존 main LSTM | LSTM 64 → Dense32 ReLU → Dense6, 21,222개 |
| Git 정보 | 제공 사본에 .git 없음, 원격 최신 commit은 확정할 수 없음 |

`src(1)/main.py`의 `save_processed_cache`는 입력 dataset 배열을 그대로 저장한다. `DataPreprocessor.fit`으로 통계는 생성하지만 cache 저장 전에 transform을 호출하지 않는다. 따라서 파일 이름의 processed라는 표현만 보고 이미 표준화됐다고 해석하면 안 된다. `src/adapters/processed_npz.py`는 이 차이를 처리하며 통계 재fit 없이 한 번 변환하고 test 배열을 읽지 않는다.

기존 raw reference adapter의 sample_id는 1 시작이고 제공 cache는 0 시작이다. RNN은 cache adapter와 원 ID를 사용한다. 서로 다른 ID 체계의 예측 파일을 행 번호만 보고 직접 합치면 대응이 틀릴 수 있다.

계획서의 SimpleRNN 기준은 은닉 32 → Dense6, 1,542개다. main의 LSTM은 계획서 기본 LSTM 32 → Dense6, 5,574개보다 큰 구조다. 이번 비교에서는 실제 main 구조를 명시하고 파라미터 규모를 함께 보고한다.

RNN 연결에는 `src/models/rnn.py`와 registry 추가가 필요했다. 비교용 CNN도 registry에 없었으므로 계획서 구조 그림의 시간축 Conv1D·Pool2·GAP·Dense32·Dropout0.2·Dense6를 추가했다. 공통 trainer, cache adapter와 saved bundle verifier는 재사용했다.

확인 수치의 원자료는 `reports/rnn/dataset_audit.json`이며 이후 실제 실행 검증은 각 run의 metadata·fresh_process_verification과 `reports/rnn/tests.xml`에서 확인한다. 원본 Downloads 자료는 수정하지 않았다.
