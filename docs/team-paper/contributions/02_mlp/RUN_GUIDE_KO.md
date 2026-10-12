# 한윤섭 MLP 실행·저장 모델 사용

명령은 저장소 루트 `smartphone-har`에서 PowerShell로 실행한다. Git clone에는 코드·문서·실행 증빙이 있고 가상환경·데이터·학습 모델은 포함되지 않는다. **[전체 Release ZIP](https://github.com/kara320090/smartphone-har/releases/download/yoonseop-mlp-v1/yoonseop-mlp-v1-full.zip)**에는 `data/processed_data.npz`, `data/preprocess.npz`, `runs/mlp_yoonseop/`의 세 저장 모델과 모든 실행 파일이 있다. 표준화 적용 전 float32 구간을 입력해야 한다.

## 전체 모델·데이터 받기

브라우저에서 ZIP을 내려받아 새 폴더에 풀거나, 다음처럼 새 작업 폴더를 만들 수 있다. SHA-256은 [Release](https://github.com/kara320090/smartphone-har/releases/tag/yoonseop-mlp-v1)의 `SHA256SUMS.txt`와 대조한다.

```powershell
Invoke-WebRequest 'https://github.com/kara320090/smartphone-har/releases/download/yoonseop-mlp-v1/yoonseop-mlp-v1-full.zip' -OutFile yoonseop-mlp-v1-full.zip
Expand-Archive -LiteralPath yoonseop-mlp-v1-full.zip -DestinationPath mlp-release-v1
Set-Location mlp-release-v1/smartphone-har
python tools/verify_mlp_release.py --directory .
```

이 검증은 표준 라이브러리만 사용해 manifest에 적힌 모든 파일의 해시를 대조한다. 학습·재로딩을 실행하려면 아래 라이브러리 설치가 필요하다.

## 현재 결과만 확인

`reports/mlp_yoonseop/summary.json`, `runs.csv`와 `docs/team-paper/contributions/02_mlp/contribution.md`를 열면 된다. [run_evidence](../../../../reports/mlp_yoonseop/run_evidence/)의 seed별 CSV·JSON은 GitHub에서도 바로 읽을 수 있고, 저장 모델·원래 실행 폴더는 Release ZIP에 있다.

## 새 환경 준비

Python 3.12의 새 가상환경을 만들 때는 다음처럼 설치한다. ZIP에 가상환경은 포함하지 않는다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

이번 실행 환경에서 임시 폴더·캐시 권한을 해결한 설정이다. 모델 저장·재로딩 및 검사 전에 적용한다. suite 스크립트는 자식 프로세스에 해당 경로를 자동 지정한다.

```powershell
New-Item -ItemType Directory -Force work\mlp_tmp,work\mplconfig,work\keras | Out-Null
$env:TEMP = (Resolve-Path work\mlp_tmp).Path
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = (Resolve-Path work\mplconfig).Path
$env:KERAS_HOME = (Resolve-Path work\keras).Path
$env:TF_CPP_MIN_LOG_LEVEL = '2'
```

## 소량 연결 검사·정식 3회

```powershell
.\.venv\Scripts\python.exe scripts/run_mlp_suite.py --runs-dir runs/mlp_smoke_new --smoke --seeds 2026
.\.venv\Scripts\python.exe scripts/run_mlp_suite.py --runs-dir runs/mlp_new
```

원본 데이터가 다른 위치면 `--processed-data 'C:/경로/processed_data.npz' --preprocess-stats data/preprocess.npz`를 추가한다. 데이터·통계 해시가 다르면 명령이 거부된다. 같은 출력 폴더의 미완료 실행은 보존하며, 재시도는 새 폴더를 사용한다. 완료 실행 재사용 시 설정을 대조하고 별도 프로세스 검증을 다시 수행한다. 기본 실행은 최고 seed를 고르지 않고 세 개 모두 실행한다.

## 지표·그림·추론·검사 재생성

```powershell
.\.venv\Scripts\python.exe -m src.mlp_analysis --runs-dir runs/mlp_new --output-dir reports/mlp_new
.\.venv\Scripts\python.exe -m src.mlp_inference --run-directory runs/mlp_yoonseop/M01_seed2026 --processed-data data/processed_data.npz --count 5 --output reports/mlp_yoonseop/prediction_demo.json
.\.venv\Scripts\python.exe -m pytest -q --basetemp=work/pytest_mlp_new
```

분석은 원래 1,775개 validation 전체, 기존 예측·지표, 저장 모델·전처리, 별도 프로세스 검증 파일을 대조한다. test는 평가하지 않는다. 위 `prediction_demo`는 validation 앞 5개의 사용 예시이며 독립 성능 평가가 아니다.

자신의 구간을 쓰려면 `--processed-data` 대신 `--input-npy windows.npy`를 사용한다. 배열은 `(N,128,9)` float32이며 채널 순서와 단위가 기존 센서 자료와 같아야 한다. 표준화는 저장 bundle이 한 번 적용한다. 새 스마트폰·새 센서에서의 정확도를 검증한 기능은 아니다.

## 하나의 모델 재로딩만 확인

```powershell
.\.venv\Scripts\python.exe -m src.verify_bundle --processed-data data/processed_data.npz --preprocess-stats data/preprocess.npz --run-directory runs/mlp_yoonseop/M01_seed2026 --report reports/mlp_reload_new.json
```

## 원본 통계가 없을 때 복원

제출 ZIP에는 원본 통계가 이미 있다. 별도의 clone에서 누락된 경우에만 공개 릴리스를 내려받아 다음을 실행한다. script는 ZIP·통계의 고정 SHA-256을 모두 검증하며 기존 통계를 다시 fit하지 않는다.

```powershell
New-Item -ItemType Directory -Force work\sources | Out-Null
Invoke-WebRequest 'https://github.com/kara320090/smartphone-har/releases/download/bongheon-lstm-v1/bongheon-lstm-v1-full.zip' -OutFile work/sources/bongheon-lstm-v1-full.zip
.\.venv\Scripts\python.exe scripts/prepare_mlp_inputs.py --source-bundle work/sources/bongheon-lstm-v1-full.zip
```

## 팀 인계

각 seed의 실행 폴더는 `model.keras`, `preprocess.npz`, `config.json`, `split.json`, `metadata.json`, `history.csv`, `validation_predictions.csv`, `metrics.json`, `reload_check.json`, `initial_gradient_check.json`, `fresh_process_verification.json`, `status.json`을 함께 전달한다. 파일을 서로 다른 seed·데이터 버전의 것으로 섞지 않는다.

기반 Git 커밋은 원격 소스이고 학습 코드의 버전은 metadata.code_sha256·protocol_freeze.json·frozen_source로 확인한다. Git 저장소의 `run_evidence`는 원본 실행 폴더의 설정·예측·지표·검증 파일을 바이트 그대로 복사한 것이며, 모델과 전체 데이터는 Release ZIP으로 제공한다. 기존 .gitignore의 data/·.venv/·runs/ 규칙을 유지했다. Release의 `MLP_RELEASE_MANIFEST.json`에는 모든 인계 파일의 SHA-256과 배포 대상 커밋을 기록한다. 공식 test·학회 제출은 이번 배포에 포함되지 않는다.
