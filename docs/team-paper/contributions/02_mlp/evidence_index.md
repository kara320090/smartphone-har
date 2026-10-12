# MLP 주장과 실행 근거

기준 경로: 저장소 루트. 정식 실행은 M01_seed2026·2027·2028의 세 개뿐이다. 저장소에서 확인할 실행 증빙은 [run_evidence](../../../../reports/mlp_yoonseop/run_evidence/), 원본 `runs/`·저장 모델·데이터는 [Release ZIP](https://github.com/kara320090/smartphone-har/releases/download/yoonseop-mlp-v1/yoonseop-mlp-v1-full.zip)에 있다. 두 위치의 공통 파일은 해시가 일치한다.

| 주장 | 원자료·재현 경로 |
|---|---|
| 새 로컬 코드 버전 고정 | [protocol_freeze.json](../../../../reports/mlp_yoonseop/protocol_freeze.json), [고정 소스](../../../../reports/mlp_yoonseop/frozen_source/) |
| 데이터·통계는 기존 팀 자료와 일치 | [dataset_audit.json](../../../../reports/mlp_yoonseop/dataset_audit.json), [통계 복원 스크립트](../../../../scripts/prepare_mlp_inputs.py), 각 run의 metadata.data_provenance |
| 입력·라벨·사람·ID 대응과 분할 | 각 run의 split.json·validation_predictions.csv, loader의 row_binding SHA-256 |
| 156,230 파라미터·층별 구조 | [model_structure.csv](../../../../reports/mlp_yoonseop/model_structure.csv), [모델 코드](../../../../src/models/mlp.py), 각 run/model.keras |
| Macro F1 0.8646 ± 0.0125, Accuracy 평균 86.67%, SD 1.17%p | [runs.csv](../../../../reports/mlp_yoonseop/runs.csv), [summary.json](../../../../reports/mlp_yoonseop/summary.json), 정식 validation_predictions.csv·metrics.json |
| 최선·종료 epoch 및 과적합 양상 | 각 run/history.csv, [learning_curves.png](../../../../reports/mlp_yoonseop/learning_curves.png), [restored fit 지표](../../../../reports/mlp_yoonseop/M01_seed2026_restored_fit_metrics.json) |
| 클래스별 성능·오분류 | [class_metrics.csv](../../../../reports/mlp_yoonseop/class_metrics.csv), [class_metrics_summary.csv](../../../../reports/mlp_yoonseop/class_metrics_summary.csv), seed별 confusion.json |
| 사람별 성능 | [subject_metrics.csv](../../../../reports/mlp_yoonseop/subject_metrics.csv), [subject_metrics_summary.csv](../../../../reports/mlp_yoonseop/subject_metrics_summary.csv) |
| 세 seed 공통 오분류 173개 | [error_consensus.csv](../../../../reports/mlp_yoonseop/error_consensus.csv)의 wrong_seed_count=3, [errors.csv](../../../../reports/mlp_yoonseop/errors.csv) |
| 저장 전후 전체 validation 대조 | 각 run의 fresh_process_verification.json, [verification.json](../../../../reports/mlp_yoonseop/verification.json), `src.verify_bundle` |
| 48개 코드 검사 통과 | [pytest.log](../../../../reports/mlp_yoonseop/pytest.log), [pytest.xml](../../../../reports/mlp_yoonseop/pytest.xml) |
| 잘못된 7종 입력 거부·평가 모드 반복 일치 | verification.json, [추론 코드](../../../../src/mlp_inference.py), [추가 인계 검사](../../../../tests/test_mlp_handoff.py) |
| 최초 환경 오류와 수정·재검증 | pytest_initial_environment_error.log/xml, `runs/mlp_yoonseop_smoke/` 중단 기록, `runs/mlp_yoonseop_smoke_fixed/` 완료 기록 |
| 재현 환경 | 각 run/metadata.json, [requirements-resolved.txt](../../../../reports/mlp_yoonseop/requirements-resolved.txt), requirements.txt |
| test 예측·평가 미실행 | metrics.json·summary.json·verification.json의 test_evaluated=false. 최초 규격 점검에서 test shape/dtype·사람 목록을 읽은 사실은 dataset_audit에 별도 기록 |
| 배포 전 전체 내용·파일 점검 | [publication_audit.json](../../../../reports/mlp_yoonseop/publication_audit.json), [Release 구성·검증 안내](../../../../reports/mlp_yoonseop/README.md) |

CSV의 seed SD는 같은 사람 분할에서의 학습 변동이다. 클래스·사람 support는 seed당 표본 수이며, 서로 같은 표본의 세 예측을 독립 표본처럼 합산하지 않는다. smoke 점수와 과거 이봉헌 MLP 점수는 정식 평균에 포함하지 않았다.
