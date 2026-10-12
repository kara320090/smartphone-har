# CNN 정식 TensorFlow 실행 및 결과 검증 안내

## 실행 상태

현재 C01 1D CNN 모델의 **정식 TensorFlow 학습 3회 완료** 및 **새 프로세스 모델 재검증 완료**가 확인된 상태입니다. 공식 test 미평가. 이 문서는 재현용 명령과 결과 경로를 정리합니다.

## 학습 조건 및 모델

- 입력 `(128, 9)`, 6개 행동, 1D CNN 85,574개 학습 파라미터.
- 훈련 5,577, 검증 1,775; 고정된 피험자 비중첩 분할 S0.
- seed 2026, 2027, 2028 / Adam 0.001 / batch 64 / 최대 40 epoch / val_loss EarlyStopping patience 6.

## 재현 명령 (Windows PowerShell)

레포지토리 루트에서 실행합니다. `data/processed_data.npz`는 팀에서 별도로 전달받아야 하며 Git에는 넣지 않습니다.

```powershell
python scripts\run_cnn_suite.py --processed-data data\processed_data.npz --runs-dir runs\cnn_retry
python scripts\cnn_summarize.py --runs-dir runs\cnn_retry --output-dir reports\cnn
python scripts\cnn_publish.py --runs-dir runs\cnn_retry --reports-dir reports\cnn --replace
```

같은 결과 폴더가 이미 있으면 재학습 시도하지 마세요. 재현 실험은 새로운 `--runs-dir`을 사용해 기존 증거를 보존해야 합니다. `--replace`는 검증 완료 원본에서 Git 추적용 결과를 다시 발행할 때만 사용합니다.

## 생성 결과

- `artifacts/cnn_tensorflow/C01_seedXXXX/model.keras`: seed별 학습 모델.
- `artifacts/cnn_tensorflow/C01_seedXXXX/fresh_process_verification.json`: 독립 프로세스 검증.
- `artifacts/cnn_tensorflow/manifest.json`: SHA-256 해시.
- `reports/cnn/cnn_summary.json`, `cnn_runs.csv`: 평가 통계.
- `reports/cnn/cnn_learning_curves.png`, `cnn_confusion.png`: 시각화.
- `docs/team-paper/contributions/03_cnn/final_tensorflow_results.md`: 정식 평가 요약.

## 확정된 Validation 통계

| Seed | Accuracy | Macro F1 |
|---|---:|---:|
| 2026 | 0.8755 | 0.8742 |
| 2027 | 0.8969 | 0.8972 |
| 2028 | 0.8941 | 0.8952 |
| **평균 ± 표본 SD** | **0.8888 ± 0.0116** | **0.8889 ± 0.0127** |

이는 test 점수가 아니며, 과거 Keras/Torch 백엔드의 예비 측정 결과와 혼합해서는 안 됩니다.
