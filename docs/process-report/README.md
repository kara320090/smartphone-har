# 이봉헌 구현 과정 심화 자료

오류 재현·수정과 실험 설계를 함께 보강한 후속 자료다. 실제 입력 계약 결함을 수정하고, 7조건×3seed 및 초기값 일치 대조 3회로 **새 학습 24회**를 수행했다. 최초 v1 실험과 구분한다.

## 읽는 순서

1. **심화 보고서 21쪽**으로 문제·원인·수정·검증·한계를 확인한다.
2. **PPT 20장**은 본문 13장과 보충 7장이다. 모든 슬라이드에 발표 노트가 있고 표 7개와 차트 2개를 편집할 수 있다.
3. **질의응답 34개**에서 자신의 설명을 점검하고 **과정 기록**으로 실제 작업 순서를 확인한다.

| 자료 | 파일 |
|---|---|
| 심화 Word 보고서 | [보고서](bongheon-process-report.docx) |
| 심화 발표 PPT | [발표 자료](bongheon-process-presentation.pptx) |
| 보고서 MD 원문 | [보고서 원문](bongheon-process-report.md) |
| 장별 원고와 4분 축약 경로 | [발표 원고](speaker-notes.md) |
| 확장 질의응답 | [34개 질문과 답변](DEFENSE_QA_KO.md) |
| 실제 과정과 판단 | [작업 과정](PROCESS_LEARNING_LOG_KO.md) |

전체 발표는 약 10–12분을 기준으로 작성했다. 4분 경로는 1→3→4→6→7→9→13이며 원고의 축약 문단을 사용한다. 실제 말하기 속도에 맞춰 연습한다.

## 확인한 결과

신규 회귀 검사 16개는 수정 전 15실패·1통과, 수정 후 기존 8개 포함 총 24통과다. ReLU의 작은 네트워크 46파라미터를 세 h 값에서 138회 비교했다. seed 2026의 기존 7조건 예측은 재학습 후 동일했고, 새 저장 모델 24개도 원본 검증 입력에서 다시 예측했다.

| 조건 | Macro F1 평균 | 표본 SD |
|---|---:|---:|
| E02 | 0.9046 | 0.0123 |
| E03 | 0.8635 | 0.0140 |
| E07 | 0.9034 | 0.0160 |
| E08 | 0.8690 | 0.0327 |
| E09 | 0.8931 | 0.0178 |
| E10 | 0.9021 | 0.0106 |
| E11 | 0.8771 | 0.0209 |
| E08W | 0.8740 | 0.0321 |

E08W는 초기 Dense 가중치와 편향을 E02에 맞춘 Dropout 0.3 대조다. SD는 같은 검증 사람 5명에서의 학습 seed 변동이다. 모집단 신뢰구간이나 통계적 우월성으로 해석하지 않는다. 공식 test는 읽거나 평가하지 않았다.

## 실행 근거와 이전 자료

[실제 수치와 실행별 기록](../../reports/process_audit/) · [반복 계획](../PROCESS_AUDIT_PLAN_KO.md) · [대조 추가 계획](../PROCESS_AUDIT_AMENDMENT_KO.md) · [이전 구현 설명과 v1 보고서](../personal-report/README.md)

[v2 전체 코드·모델·문서 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-process-v2)에는 최초 7개와 새 24개 모델이 들어 있다. Git에는 요약 근거를, 릴리스 ZIP에는 모델·전처리·표본별 예측을 보관한다. 원 데이터와 가상환경은 포함하지 않는다.

## 재현 안내

Python 3.11 환경과 requirements.txt를 설치하고 공식 UCI 자료를 준비한다. 아래는 저장소 루트에서 실행하는 PowerShell 예다. 재실행은 별도 복사본에서 수행한다. 기존 runs 폴더는 덮어쓰지 않는다.

```powershell
$env:TF_ENABLE_ONEDNN_OPTS='0'
$env:TF_CPP_MIN_LOG_LEVEL='2'
python -m pytest tests -q
python analysis/process_audit/contract_probe.py --phase reproduced-after --output work/reproduced-contracts.json
python analysis/process_audit/math_controls.py
python analysis/process_audit/initialization_control.py
python analysis/process_audit/run_repeats.py 'data/raw/UCI HAR Dataset'
foreach ($harSeed in 2026,2027,2028) {
    $env:PYTHONHASHSEED=[string]$harSeed
    python analysis/process_audit/run_matched_dropout.py --seed $harSeed --data-root 'data/raw/UCI HAR Dataset'
}
python -m src.verify_bundle --data-root 'data/raw/UCI HAR Dataset' --runs-dir runs/process_repeats --report work/reload-repeats.json
python -m src.verify_bundle --data-root 'data/raw/UCI HAR Dataset' --runs-dir runs/process_matched --report work/reload-matched.json
python analysis/process_audit/summarize.py
```

`math_controls.py`와 `initialization_control.py`는 reports/process_audit의 해당 JSON을 새로 쓴다. 보존 기록을 유지하려면 별도 복사본에서 실행한다. summarize.py는 새 24개 실행뿐 아니라 v1의 `runs/formal/` 7개 예측도 필요하다. v2 전체 ZIP에 모두 포함했다. 21회 드라이버의 로그는 현재 스크립트에 정한 경로에 저장되며, 실행 전에 경로를 확인한다.

수정 전 재현은 `7aec3ac`의 src/contracts.py와 신규 회귀 검사를 분리된 복사본에서 함께 실행했던 기록이다. 현재 수정본에 `--phase before`라는 이름만 주면 수정 전 코드로 돌아가는 것은 아니다. 실제 소스 SHA와 before/after 기록을 대조한다.

저장 모델은 `compile=False`로 추론·평가하는 용도다. 정확한 학습 재개, 다른 장치의 비트 단위 재현, 공식 test 성능, 팀 통합·시연은 별도 검증 범위다.

자료에는 AI 보조로 수행한 코드·실험·문서 정리가 포함된다. 실제 실행한 검증과 의도적 대조를 구분했으며, 발표자가 검토·이해한 범위에 맞춰 설명한다.
