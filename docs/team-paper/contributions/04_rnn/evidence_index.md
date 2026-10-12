# 유재윤 RNN 근거 목록

경로는 저장소 루트 기준이다.

| 주장·결과 | 근거 |
|---|---|
| 상태 초기화·32/64 은닉·출력 구조 | src/models/rnn.py, tests/test_rnn.py, STRUCTURE_KO.md |
| 데이터·한 번 표준화·피험자 분할 | src/adapters/processed_npz.py, reports/rnn/dataset_audit.json, run_evidence 각 metadata·split |
| 공통 조건·9회 설정·R03 초기 가중치 | docs/RNN_TRAINING_PROTOCOL_KO.md, configs/R01~R03.json, run_evidence 각 config·initialization |
| 3개 조건의 평균·표본 표준편차 | reports/rnn/summary.csv, runs.csv, analysis.json |
| 학습 곡선·같은 에포크 차이 | reports/rnn/learning_diagnosis.csv, run_evidence 각 history.csv, figures/*_history.png |
| 행동별·피험자별 성능·전체 오류 | reports/rnn/class_metrics.csv, subject_metrics.csv, errors.csv, figures/*_confusion.png |
| 오류·정분류 3쌍 | reports/rnn/error_success_cases.json, error_success_waveforms.npz, figures/error_success_pair_*.png |
| 수식·입력·상태·새 실행 확인 | reports/rnn/math_verification.json, integration_tests.xml, delivery_verification.json |
| 노트북 실행 | notebooks/04_rnn_yu_jaeyun.ipynb, reports/rnn/notebook_execution.json |
| 모델·통계·입력 데이터 무결성 | Release의 RNN_RELEASE_MANIFEST.json, SHA256SUMS.txt, run_evidence 각 fresh_process_verification.json |
| 과거 코드 해시와 게시 소스 차이 | reports/rnn/SOURCE_PROVENANCE_KO.md |

공식 test는 미평가다. 팀 전체 비교 자료는 이번 RNN 전용 게시 범위에 포함하지 않았다.
