# RNN 주장과 근거

| 주장 | 근거 |
|---|---|
| 제공 NPZ·저장 통계 일치와 비누수 | reports/rnn/dataset_audit.json |
| 1,542개 기본 RNN과 상태 초기화 | src/models/rnn.py · tests/test_rnn.py |
| 사전 후보와 선택 규칙 | docs/RNN_TRAINING_PROTOCOL_KO.md · configs/R01–R03.json |
| seed별 지표 재계산 | reports/rnn/runs.csv · runs/rnn/*/validation_predictions.csv |
| 클래스·사람별 성능과 오류 | reports/rnn/class_metrics.csv · subject_metrics.csv · errors.csv |
| 계산 정확성 | reports/rnn/math_verification.json |
| 전체 validation 재로딩 | runs/rnn/*/fresh_process_verification.json |
| 초기 고정 배치 진단 | runs/rnn/R*/early_gradient_probes.json |
| 네 모델 동일 입력 보조 비교 | reports/rnn/four_model_comparison.csv · paired_differences.csv |
| 실행 코드 테스트 | reports/rnn/tests.xml |
