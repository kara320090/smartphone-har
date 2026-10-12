# 유재윤 RNN 실행·결과 안내

팀 인계 원고는 docs/team-paper/contributions/04_rnn/, 상세 결과는 docs/rnn-report/, 원자료와 그래프는 reports/rnn/에 있다. 기준 R01과 튜닝 선택 R02를 구분한다. 공식 test는 미평가다.

## 파일을 받는 방법

Git에는 코드·문서·노트북·그림·CSV·18회 숫자 증빙을 보관한다. 전체 모델·초기 가중치·통계·데이터·정확한 당시 코드·smoke 기록은 `jaeyoon-rnn-v1-full.zip` RNN Release 자산이다. 게시자는 main 반영 후 tag `jaeyoon-rnn-v1`로 Release를 생성해 해당 ZIP과 SHA256SUMS.txt를 첨부한다. 이 안내 작성 시점에는 RNN Release가 아직 게시되지 않았다.

## 현재 main에서 RNN을 새로 학습

Release ZIP을 별도 폴더에 풀고 그 안의 data/processed_data.npz와 data/preprocess.npz를 현재 프로젝트 data/에 놓는다. 같은 데이터가 이미 있다면 아래 해시가 일치하는지 확인하고 사용한다. 두 파일은 기존 팀 데이터이며 표준화 전 배열과 fit 통계다. 입력을 이중 표준화하지 않는다.

- processed_data.npz SHA-256: 00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459
- preprocess.npz SHA-256: 42b1b895d420a21ed7cce3b3636a4cbed96350953b27057c0670db9062d79fb2

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_rnn.py
.\.venv\Scripts\python.exe scripts/run_rnn_suite.py --runs-dir runs/rnn-new
.\.venv\Scripts\python.exe -m src.rnn_analysis --runs-dir runs/rnn-new --out reports/rnn-new
```

새 suite는 RNN 9회만 실행하며 `src.rnn_train`을 사용한다. R03은 같은 seed R02의 학습 전 가중치를 복사하므로 suite로 순서대로 실행한다. 다른 모델의 기존 공통 trainer·CNN·M01/C01 설정은 유지했다. 최신 통합 코드로 새 실행하면 코드 해시가 이전 실행과 달라진다. 이전 수치를 최신 소스로 실행한 결과라고 바꾸어 쓰지 않는다.

## 기존 전체 18회 결과·노트북 재현

Release ZIP의 `jaeyoon-rnn-runtime` 폴더는 당시 코드와 데이터·runs/rnn·reports/rnn·notebook을 같은 상대 경로에 둔 독립 실행 묶음이다. 그 폴더를 작업 디렉터리로 삼고 의존성을 설치한 뒤 원래 `scripts/run_rnn_suite.py`, `src.rnn_analysis`, `reports/rnn/reproduce_diagnosis.py`와 notebook을 사용한다. 새로 학습할 때는 새 runs 폴더를 지정한다. 현재 main 위에 이 폴더의 전체 src/configs를 덮어쓰지 않는다.

Git만 받은 상태에서도 `python -m src.rnn_analysis --runs-dir reports/rnn/run_evidence --out reports/rnn-recomputed`로 저장 예측의 수치·그림을 재계산할 수 있다. 모델·원 센서 파형이 필요한 notebook은 Release 폴더에서 실행한다. Git에 올라간 notebook 출력은 당시 실행 결과이며 현재 main으로 실행한 결과를 뜻하지 않는다.

현재 main의 CNN C01과 보조 비교용 C01은 구조가 다르다. reports/rnn/frozen_source의 C01·M01을 최상위 configs/에 복사하지 않는다.
