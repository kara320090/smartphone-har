# 1D CNN 정식 실험 근거 색인

| 주장 | 확인 파일 | 범위 |
|---|---|---|
| 128×9 입력, 6 logits 출력, 파라미터 85,574개 | `src/models/cnn.py`, `configs/C01.json`, `tests/test_cnn.py`, `artifacts/cnn_tensorflow/C01_seed2026/metadata.json` | CNN 정식 설정 및 저장 모델 기록 |
| TensorFlow C01 3 seed 정식 학습 | `reports/cnn/cnn_summary.json`, `reports/cnn/cnn_runs.csv` | seed 2026·2027·2028 |
| Accuracy 및 Macro F1 | `reports/cnn/cnn_summary.json`, `docs/team-paper/contributions/03_cnn/final_tensorflow_results.md` | 고정 피험자 validation 1,775개 |
| 독립 프로세스에서 저장 모델 재검증 | `artifacts/cnn_tensorflow/C01_seedXXXX/fresh_process_verification.json`, `reload_check.json` | seed별 모델 재검증 기록 |
| 클래스 및 피험자별 분석 | `reports/cnn/*_per_class.csv`, `reports/cnn/cnn_per_subject.csv`, `reports/cnn/cnn_confusion.png` | validation |
| 학습 동향 | `reports/cnn/cnn_learning_curves.png`, `artifacts/cnn_tensorflow/C01_seedXXXX/history.csv` | loss 학습 곡선 |
| 산출물 무결성 | `artifacts/cnn_tensorflow/manifest.json` | 파일별 SHA-256 |
| 공식 test 미평가 | `reports/cnn/cnn_summary.json`, `artifacts/cnn_tensorflow/manifest.json` | test 결과 미포함 |

`data/`와 `runs/`는 Git에서 제외된 원본 데이터/중간 실행 폴더입니다. 정식 결과는 `reports/cnn/`, `artifacts/cnn_tensorflow/`에 보관합니다.
