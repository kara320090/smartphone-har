# 네 모델 예측 재계산 원자료

2026년 10월 13일 main `9d1da37aaac538e0ee6fd91d2a8335795d2e469b`에서 MLP M01, CNN C01, RNN R01, LSTM L01의 기준 12회와 RNN R02의 추가 3회를 검사했다. 평가 대상은 동일한 개발 validation 1,775개다. 공식 test는 미평가다.

- `baseline_summary.csv`: 기준 네 모델, 각각 세 seed의 평균과 표본 SD. 팀의 최종 대표 선정 확정을 대신하지 않는다.
- `supplementary_summary.csv`: RNN 담당자의 개인 튜닝 선택 R02. 기준 표와 섞지 않았다.
- `runs_recomputed.csv`: 예측에서 재계산한 실행별 지표와 모델·Dropout·환경 정보. 학습 시간 순위는 포함하지 않았다.
- `paired_seed_differences.csv`: 같은 seed에서 A−B 방향의 차이. 세 쌍을 독립 사람 모집단 표본으로 해석하지 않는다.
- `subject_metrics.csv`: 사람별 support, Accuracy와 여섯 클래스 Macro F1.
- `audit.json`: 데이터·통계·실제 분할·ID/사람/정답 순서 검사, 게시 지표 대조, 근거 파일 해시. 모델 재로딩은 기존 검증 기록을 확인했으며 이 감사에서 전 모델을 재로딩하지 않았다.
- `test_verification.json`: 병합된 main에서 실행한 68개 코드 테스트의 결과.

[통합 비교·그림·논문 초안과 남은 확정 사항](../../docs/team-paper/integrated-20261013/README.md)

저장소 루트에서 LSTM Release의 `runs/lstm`을 준비한 뒤 `python tools/audit_team_results.py --lstm-runs runs/lstm --output reports/team_comparison`으로 재계산한다. 다른 담당자 예측은 저장소에 있다. 재실행 시 새 `--output` 경로를 권장한다. 이 명령은 학습하거나 test를 평가하지 않는다.
