# CNN 파트 최종 인계

**상태:** TensorFlow 백엔드에서 C01 seed 2026/2027/2028 학습·검증 및 독립 새 프로세스 재검증 완료. 공식 test 결과는 없음.

- `contribution.md`: 팀 논문/보고서에 반영할 CNN 담당 원고(정식 TensorFlow validation 결과).
- `final_tensorflow_results.md`: 실험 결과를 자동 집계한 원본 요약(사용자 PC에서 생성됨).
- `evidence_index.md`: 결과와 재현성 증거의 경로.
- `reports/cnn/`: 정식 검증 지표·학습 곡선·혼동행렬.
- `artifacts/cnn_tensorflow/`: 저장 모델·검증 기록·SHA-256 manifest.
- `docs/CNN_EXECUTION_KO.md`: 재현 절차.

**중요:** Accuracy 0.8888 ± 0.0116, Macro F1 0.8889 ± 0.0127은 1,775개 validation 표본 기준입니다. test 결과로 표기하지 마세요. 이전 Torch 예비 값은 이 보고서에 병합하지 않습니다.
