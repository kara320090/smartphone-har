# RNN 인계 근거 목록

아래 경로는 전체 프로젝트 루트 기준이다. 대용량 데이터·모델은 복제하지 않고 원래 실행 묶음을 함께 인계한다.

| 확인 내용 | 원자료 |
|---|---|
| 데이터 규격·분할·fit 통계와 한 번 표준화 | reports/rnn/dataset_audit.json, data/processed_data.npz, data/preprocess.npz |
| RNN 구조·상태 초기화 | src/models/rnn.py, tests/test_rnn.py, STRUCTURE_KO.md |
| 학습 전에 정한 조건 | docs/RNN_TRAINING_PROTOCOL_KO.md, configs/R01.json~R03.json |
| 기준 세 seed | runs/rnn/R01_seed2026, R01_seed2027, R01_seed2028 |
| 보조 튜닝 여섯 실행 | runs/rnn/R02_seed2026~2028, R03_seed2026~2028 |
| 실행별 설정·장치·코드 해시·모델·통계 | 각 실행의 config.json, metadata.json, model.keras, preprocess.npz |
| epoch별 곡선·선택과 종료 | 각 실행의 history.csv, metrics.json; reports/rnn/figures/*_history.png |
| 저장 예측과 전체 검증 재로딩 | 각 실행의 validation_predictions.csv, fresh_process_verification.json |
| 모든 seed 지표·표본 SD | reports/rnn/runs.csv, summary.csv |
| 클래스·사람별 결과·오류 | reports/rnn/class_metrics.csv, subject_metrics.csv, errors.csv |
| 혼동행렬 | reports/rnn/*_confusion.json, reports/rnn/figures/*_confusion.png |
| 수치미분·초기 진단 | reports/rnn/math_verification.json, 각 RNN 실행의 early_gradient_probes.json |
| 비교용 보조 재실행 | reports/rnn/four_model_comparison.csv, paired_differences.csv |
| 코드 검사·노트북 실행 기록 | reports/rnn/tests.xml, notebook_execution.json |
| 이번 요구사항 대조 | docs/rnn-report/TEAM_REQUIREMENTS_AUDIT_KO.md, reports/rnn/team_requirements_audit.json |

validation에서 선택한 R02의 점수를 독립 test 성능으로 표현하지 않는다. 초기 고정 배치 norm을 실제 학습 배치 또는 시점별 기울기 측정으로 표현하지 않는다.
