# 한윤섭 MLP 실행 증빙과 배포 구성

M01 seed 2026·2027·2028의 validation 결과는 Macro F1 **0.8646 ± 0.0125**, Accuracy 평균 **86.67%**, Accuracy의 seed 표본 SD **1.17%p**다. 공식 test는 미평가다.

| 확인할 내용 | 자료 |
|---|---|
| 세 seed 설정·학습 기록·예측·지표·재로딩 | [2026](run_evidence/M01_seed2026/), [2027](run_evidence/M01_seed2027/), [2028](run_evidence/M01_seed2028/) |
| 모델·원본 실행 폴더·전처리 데이터·통계 | [전체 Release ZIP](https://github.com/kara320090/smartphone-har/releases/download/yoonseop-mlp-v1/yoonseop-mlp-v1-full.zip) |
| 평균·표본 SD·실행별 환경과 코드 버전 | [summary.json](summary.json), [runs.csv](runs.csv) |
| 클래스·사람별 성능·반복 오분류 | [class_metrics_summary.csv](class_metrics_summary.csv), [subject_metrics_summary.csv](subject_metrics_summary.csv), [error_consensus.csv](error_consensus.csv) |
| 학습 전에 고정한 코드·규약 | [protocol_freeze.json](protocol_freeze.json), [frozen_source](frozen_source/) |
| 소량 연결 검사·초기 환경 문제 | [smoke_evidence](smoke_evidence/), pytest_initial_environment_error.log/xml |
| 최초 48개 코드 검사·로컬 완성 확인 | pytest.log/xml, final_artifact_check.json |
| 배포 전 재검사·모델 재로딩·지표 재계산 | pytest_prepublication.log/xml, prepublication_reload.log, [publication_audit.json](publication_audit.json) |

`run_evidence`의 공통 파일은 Release ZIP의 `runs/mlp_yoonseop/M01_seed*/` 파일과 바이트·해시가 같다. Git clone에서 모델이 보이지 않는 것은 `.gitignore`의 기존 대용량 파일 제외 규칙에 따른다. Release에서 원본 전체를 받으면 새 학습 없이 저장 모델을 재로딩할 수 있다.

`frozen_source`와 `protocol_freeze.json`은 정식 학습 **이전**의 기록이며 수정하지 않았다. 학습 코드의 SHA-256은 `39152401b2cbf1ac4db720ed6401e901def0b3ea4690e84536b6450a2a1b15a0`이다. 당시의 Git 기반 커밋과 새 코드 snapshot을 구분한다. 최초 로컬 ZIP의 manifest는 `local_handoff_manifest_20261012.json`에 보존했으며, 그 안의 미푸시 표시는 당시 상태다. 최신 배포 파일은 Release의 `MLP_RELEASE_MANIFEST.json`·`SHA256SUMS.txt`로 검증한다.

완료한 MLP 실험·원고의 범위는 [개인 원고](../../docs/team-paper/contributions/02_mlp/contribution.md)에 적었다. 다른 모델의 실험이나 팀 최종 test·학회 제출을 완료한 기록은 아니다.
