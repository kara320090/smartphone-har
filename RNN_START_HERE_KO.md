# 유재윤 RNN 실행·업로드 안내

영문 이름이 필요한 경우 **Yu jaeyun**을 사용한다. 설명·원고·그래프는 한글 중심으로 작성했다. 게시 범위는 R01·R02·R03 × 세 seed의 RNN 9회다.

## 게시 위치와 구성

- 브랜치: feat/yu-jaeyun-rnn
- [RNN 전용 Release](https://github.com/kara320090/smartphone-har/releases/tag/yu-jaeyun-rnn-v1): yu-jaeyun-rnn-v1-full.zip, RNN_RELEASE_MANIFEST.json, SHA256SUMS.txt
- Git: RNN 코드·설정·실행 노트북·원고·그래프·CSV·9회 원자료
- Release ZIP: RNN 전용 실행 코드·데이터·전처리 통계·9개 저장 모델·초기 가중치·원자료·보고서

팀 인계 자료는 docs/team-paper/contributions/04_rnn/, 보고서·대본·질의응답은 docs/rnn-report/다. 현재 main에 있는 다른 담당자의 모델 자료를 유지하고 RNN 전용 진입점을 추가했다. 원고는 한글 이름을 사용한다.

## 검증 지표 재계산(Git만 필요)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.rnn_analysis --runs-dir reports/rnn/run_evidence --out reports/rnn-recomputed
```

## 저장 모델·노트북 재현

Release ZIP을 별도 폴더에 풀고 `yu-jaeyun-rnn-runtime`을 작업 디렉터리로 삼는다. 그 안에서 의존성을 설치한 뒤 다음 명령을 사용한다. 새 검증 기록은 원 실행 기록을 덮어쓰지 않는 reports/rnn-rechecks에 남긴다.

```powershell
python -m src.verify_bundle --processed-data data/processed_data.npz --preprocess-stats data/preprocess.npz --run-directory runs/rnn/R02_seed2026 --report reports/rnn-rechecks/R02_seed2026.json
python -m src.rnn_analysis --runs-dir runs/rnn --out reports/rnn
python reports/rnn/reproduce_diagnosis.py
```

노트북은 notebooks/04_rnn_yu_jaeyun.ipynb다. 실행 출력도 게시했다. `python -m src.rnn_math_checks --help`로 수식 검증 명령을 확인할 수 있다.

## 새 RNN 학습

현재 팀 저장소에서 실행할 때는 ZIP의 data/processed_data.npz·data/preprocess.npz 두 파일을 프로젝트 data/에 놓는다. ZIP의 전체 소스 폴더를 현재 팀 저장소에 덮어쓰지 않는다.

```powershell
python scripts/run_rnn_suite.py --runs-dir runs/rnn-new
python -m src.rnn_analysis --runs-dir runs/rnn-new --out reports/rnn-new
```

R03은 같은 seed R02의 학습 전 가중치를 사용하므로 suite의 실행 순서를 유지한다. 새 실행은 새 폴더로 기록한다. 저장 예측과 새 모델 확인은 검증셋만 사용하며 공식 test는 평가하지 않는다. 과거 전체 개발 폴더의 코드 해시와 게시 소스 차이는 reports/rnn/SOURCE_PROVENANCE_KO.md에 설명했다.

## 데이터 해시

- processed_data.npz: 00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459
- preprocess.npz: 42b1b895d420a21ed7cce3b3636a4cbed96350953b27057c0670db9062d79fb2

이름이 같아도 해시가 다른 데이터는 이번 결과의 입력으로 간주하지 않는다. 공식 test·팀 전체 비교·최종 통합 논문과 슬라이드는 팀 공동 결정 이후의 작업이다.
