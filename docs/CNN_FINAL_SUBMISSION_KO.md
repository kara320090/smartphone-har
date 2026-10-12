# CNN TensorFlow 정식 제출 확인

## 현재 상태

2026년 10월 12일 사용자 PC의 TensorFlow 실행에서 C01 seed 2026, 2027, 2028 모델 학습·검증 및 저장 모델의 독립 프로세스 재검증이 완료되었습니다. 제출용 발행 스크립트(`scripts/cnn_publish.py`)도 정상 완료되었다고 보고받았습니다.

**검증 평균:** Accuracy **0.8888 ± 0.0116**, Macro F1 **0.8889 ± 0.0127** (seed 간 표본 표준편차, validation 1,775개). **공식 test 미평가**.

## 팀 레포에 포함할 산출물

- CNN 구현: `src/models/cnn.py` 및 모델 등록부 `src/models/__init__.py`
- 실험 코드: `src/train.py`의 CNN 전체 validation 재로딩 검사, `configs/C01.json`, `scripts/run_cnn_suite.py`, `scripts/cnn_summarize.py`, `scripts/cnn_publish.py`, `tests/test_cnn.py`
- 결과: `reports/cnn/` 전체(표, 곡선, 혼동행렬) 및 `artifacts/cnn_tensorflow/` 전체(3개 `.keras`와 검증·무결성 자료)
- 논문용 원고: `docs/team-paper/contributions/03_cnn/` 전체

데이터 캐시 `data/processed_data.npz`와 중간 결과 `runs/cnn_retry`는 `.gitignore`로 제외하며 별도 전달 대상으로 관리합니다. 다른 팀원의 기존 파일 변경 시 최신 레포를 기준으로 차이를 검토하고 병합하십시오.

## 직접 검증

```powershell
Get-Content .\docs\team-paper\contributions\03_cnn\final_tensorflow_results.md
Get-ChildItem .\artifacts\cnn_tensorflow\C01_seed*\model.keras
Get-ChildItem .\reports\cnn\cnn_summary.json
```

공식 test는 팀 전체 모델/실험 방식을 동결한 뒤 합의하여 평가합니다.
