# 유재윤 RNN 팀 인계 자료

contribution.md는 7문단 내부 원고, RNN_COMPACT_INSERT_KO.md는 최종 공동 원고 취합용 짧은 방법 문단, RNN_COMPARISON_ROW.csv는 기준 R01의 세 seed 한 행이다. STRUCTURE_KO.md, references.md, evidence_index.md와 figures를 함께 확인한다.

전체 보고서는 ../../../rnn-report/REPORT_KO.md, 설치·재현 안내는 프로젝트 루트 RNN_START_HERE_KO.md다. 프로젝트 루트에서 `python scripts/run_rnn_suite.py --runs-dir runs/rnn-new`로 새 실행하고 `python -m src.rnn_analysis --runs-dir runs/rnn-new --out reports/rnn-new`로 집계한다. 데이터와 모델을 포함한 전체 ZIP을 인계해야 원자료 경로가 유지된다.

그림 R01_seed2026_history.png는 기준 RNN의 seed 2026 epoch별 교차엔트로피·Accuracy·Macro F1과 최저 validation loss epoch를 보여준다. 원자료는 runs/rnn/R01_seed2026/history.csv다. R01_seed2026_confusion.png는 같은 실행의 validation 혼동행렬이며 행=정답·열=예측·셀 단위=구간 수다. 원자료는 reports/rnn/R01_seed2026_confusion.json과 해당 실행의 validation_predictions.csv다. 대표 그림 두 장은 전체 seed를 대신하는 성능 추정치가 아니다.

상태는 개발 실험 완료이며 공식 test·공통 규약 확정·담당자 검토가 남아 있다. 이 파일들 자체가 학회 제출용 최종 논문은 아니다.
