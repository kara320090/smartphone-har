# 공식 test 평가를 위한 추가 대조

[고정 계획](PROTOCOL_KO.md) · [과정 기록](PROCESS_LOG_KO.md) · [코드](../../tools/confirmatory_har.py)

새 신경망 15회, 기존 신경망 12개, 선형 모델 2개를 사용한다. 기존 모델의 개발 결과를 본 뒤 추가 조건을 정했으며, 연구 전체의 사전등록이나 새 알고리즘 개발로 표현하지 않는다.

## 실행 순서

Python 환경은 기존 TensorFlow 2.21.0, Keras 3.15.1 환경이다. 입력 `processed_data.npz`와 `data/preprocess.npz`는 프로토콜에 지정한 해시와 일치해야 한다. 공식 UCI 자료는 cache와 원 행을 대조할 때 사용한다.

1. `python -m pytest tests/test_confirmatory.py -q`
2. 각 조건 FMLP3/MLP3/CNN3/RNN3/LSTM3, seed 2026/2027/2028에 대해 `python tools/confirmatory_har.py train --condition FMLP3 --seed 2026` 형태로 별도 프로세스 실행.
3. `python tools/confirmatory_har.py inventory`
4. inventory의 모든 run에 대해 `verify-dev --run RUN_ID`, 이어서 `benchmark --run RUN_ID`를 새 프로세스에서 순차 실행.
5. `python tools/confirmatory_har.py freeze`, lock·개발 결과를 커밋해 고정.
6. `source-check`, 모든 run의 `test --run RUN_ID`, 새 프로세스 `replay --run RUN_ID`, 마지막 `audit`.

반복 실행 도구는 `python scripts/confirmatory_jobs.py PHASE`이다. 학습·시간 측정은 항상 한 프로세스씩 순차 진행한다. 시간 측정을 하지 않는 test 예측과 재예측은 `--workers 3`으로 서로 다른 모델의 독립 프로세스 세 개를 병행할 수 있다. 모델·입력·평가 규칙은 같으며 결과 디렉터리는 각각 분리한다.

CLI의 각 하위 명령은 `python tools/confirmatory_har.py` 뒤에 붙인다. 기본 원본 데이터 경로는 프로젝트 실행 당시 바탕화면 경로이고, 원본 UCI 비교 경로는 작업공간의 `work/uci-har/UCI HAR Dataset`이다. 다른 환경에서는 `HAR_PROCESSED_DATA`, `HAR_OFFICIAL_ROOT` 환경변수로 두 경로를 지정할 수 있다. 입력 파일과 평가 코드 해시를 유지하면 원 lock으로 저장 모델을 재검증할 수 있다.

완성된 lock이 있는 실행 폴더에서는 새 학습이 금지된다. 재현 학습은 별도 작업 사본에서 시작하고 원 lock과 재현 결과를 구분한다. 이미 생성된 학습 폴더나 공식 test 첫 평가 폴더가 있으면 도구가 중단한다. 재검증은 `replay`로 수행하며 원 예측을 덮어쓰지 않는다.

## 지표 해석

- 6종과 3상태 Macro F1은 서로 다른 과제다.
- seed 표본 SD는 모델 초기화·학습 변동이며, 사람 모집단의 신뢰구간이 아니다.
- LR3와 FMLP3는 같은 특징·같은 목표를 사용하지만 분류기 용량과 최적화 방법이 다르다. 이는 정해진 두 구성의 비교다.
- 시간은 이미 준비된 센서 구간에서 출력까지이며, 센서 수집과 UCI 필터링·윈도 생성은 제외한다.
- RSS는 실행 런타임까지 포함한 프로세스 수치이고 순수 모델 메모리가 아니다. 가중치/표준화 배열 바이트와 분리한다.
- 후속 앱·주머니 착용·다른 기기·전력·수면 판정은 이번 연구 범위에 포함되지 않는다.
