# CNN 정식 TensorFlow 검증 결과 (자동 생성)

> **TensorFlow 정식 실행 및 독립 새 프로세스 재검증 확인 후 생성되었습니다.** 공식 test 결과가 아니라 고정 피험자 분할 S0의 validation 결과입니다.

## 실험 조건

- 데이터: UCI HAR, 128 시점 × 9 채널, 6개 행동.
- 학습 5,577개, 검증 1,775개, 피험자 비중첩 분할.
- 모델: 1D CNN, 85,574 학습 파라미터.
- seed: 2026, 2027, 2028.
- Adam(lr=0.001), batch=64, 최대 40 epoch, val_loss 조기 종료 patience=6.
- 백엔드: TensorFlow (학습 폴더 metadata.json의 실제 버전 참조).
- 제출한 평가 수치는 validation 기준이며 test 집합은 평가하지 않았습니다.

## 검증 결과

| Seed | Accuracy | Macro F1 | 종료 Epoch | 최저 검증 손실 Epoch |
|---|---:|---:|---:|---:|
| 2026 | 0.8755 | 0.8742 | 8 | 2 |
| 2027 | 0.8969 | 0.8972 | 7 | 1 |
| 2028 | 0.8941 | 0.8952 | 10 | 4 |
| **평균 ± 표본 SD** | **0.8888 ± 0.0116** | **0.8889 ± 0.0127** | — | — |

세 seed의 표본 표준편차는 **고정된 분할에서 초기화 seed 변화**를 나타내며 새로운 피험자 집단에 대한 통계적 신뢰구간이 아닙니다.

## 재현 및 증거

- `reports/cnn/cnn_summary.json`: 세 seed 검증 통계
- `reports/cnn/cnn_runs.csv`: 개별 실행 기록
- `reports/cnn/cnn_confusion.png`: 검증 혼동행렬
- `reports/cnn/cnn_learning_curves.png`: 학습·검증 손실
- `artifacts/cnn_tensorflow/C01_seedXXXX/`: 실제 모델·예측·평가·재검증 기록
- `artifacts/cnn_tensorflow/manifest.json`: 파일별 SHA-256

작성자 확인 없이 예비 Keras/Torch 결과와 평균을 섞어 비교하지 마십시오.
