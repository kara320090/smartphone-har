# 봉헌님이 지금 확인할 것

담당 ②의 로컬 구현과 UCI 실제 데이터 실험 7개를 마쳤다. [결과 보고서](reports/RESULTS_KO.md)와 [조건별 해석](docs/EXPERIMENT_NOTES_KO.md)을 먼저 읽으면 된다. 코드와 문서는 이 저장소에서, 학습 모델은 [v1 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1)에서 제공한다.

## 준비된 결과

- 특징 MLP, 시계열 MLP, 다른 Keras 모델도 받을 수 있는 공통 학습 함수
- E02·E03·E07·E08·E09·E10·E11의 모델, 전처리, 설정, 학습 곡선, 검증 예측
- 역전파 전체 17개 파라미터의 수치·NumPy·자동미분 비교
- TFRecord 200개 표본의 정확한 저장·읽기 확인
- 7개 저장 모델을 새 프로세스에서 원 입력부터 다시 예측한 확인 기록
- 설명 노트북 3개, 재현 환경, 팀 연결 문서, 약 4분 발표 원고

기본 특징 MLP E02는 검증 Accuracy 91.61%, macro F1 0.9138이다. 다른 조건을 포함한 7개 중 가장 높았지만, 이것은 **seed 2026의 검증 결과**다. 공식 시험 평가와 다른 담당자의 모델 비교는 팀 일정에 따라 진행한다.

## 실행 환경 준비

처음 받은 팀원은 [상세 실행 안내](docs/RUN_GUIDE_KO.md)에 따라 Python 환경과 공식 데이터를 준비한다. 아래 로컬 스크립트는 저장소의 `.venv` 또는 이 작업에서 만든 환경을 찾아 사용한다.

## 현재 PC에서 바로 실행

프로젝트 폴더의 PowerShell에서 다음 명령을 실행한다. 스크립트는 이 작업에서 만든 Python 환경과 다운로드한 데이터를 찾아 사용한다. 다른 컴퓨터로 옮겼다면 [상세 실행 안내](docs/RUN_GUIDE_KO.md)의 설치 절차부터 진행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_local.ps1 -Action smoke
```

현재 명령에만 실행 정책을 적용하며 시스템 설정을 변경하지 않는다. 작은 실제 데이터로 2 epoch 학습하고 새로운 폴더에 저장한다. 저장 모델 검증은 아래 릴리스 모델을 배치한 뒤 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_local.ps1 -Action verify
```

## 저장된 모델 사용

1. [v1 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1)에서 `smartphone-har-bongheon-complete.zip`을 받는다.
2. 압축을 풀어 나온 `smartphone-har` 폴더에서 작업한다. 이미 clone했다면 ZIP의 `runs/formal` 폴더를 clone한 저장소에 복사한다.
3. Python 환경 설치와 데이터 다운로드를 완료한 뒤 위의 `-Action verify` 명령을 실행한다.

릴리스는 모델과 전처리를 함께 제공한다. GitHub의 일반 Source code ZIP에는 Git에서 제외한 학습 모델이 없으므로 릴리스의 이름이 지정된 ZIP을 받는다.

## 팀원에게 보낼 것

코드는 저장소의 `src`, `configs`, `scripts`, `tests`, 환경 파일과 문서를 함께 공유한다. 추론 담당자에게는 `runs/formal/E02_seed2026` 등 실행 폴더를 통째로 전달한다. 학습 모델 7개는 [v1 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1)에서 공유한다. 원 데이터는 공식 다운로드 스크립트로 준비한다.

① 담당자에게는 [입력 계약](docs/HANDOFF_KO.md)을 공유하고, 데이터 모듈이 완성되면 sample_id·분할·전처리 수치가 같은지 확인한다. 각자 독립 작업을 계속하다가 이 계약에 맞춰 연결하면 된다.

## 봉헌님이 직접 해야 하는 남은 일

1. 노트북을 실행하며 입력 shape, 손실, 역전파, 조기 종료를 이해한다.
2. 팀원에게 저장물을 넘기고 그 사람의 환경에서 실제 실행되는지 확인한다.
3. 팀장으로 모듈 연결과 전체 모델 실험 일정을 맞춘다.
4. [발표 원고](docs/PRESENTATION_KO.md)를 자신의 말로 설명하고 질문에 답할 준비를 한다.

팀원 구현과의 실제 연결, 다른 사람의 실행 확인, 발표 연습은 아직 완료했다고 표시하지 않았다. 로컬에서 독립적으로 구현·측정할 수 있는 담당 업무는 코드와 근거 파일로 준비했다.
