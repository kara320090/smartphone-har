================================================================================
  스마트폰 센서 행동인식 프로젝트 - 1주차 개발용 데이터 아웃풋 안내 (README.txt)
  담당: ① 데이터 분석 및 전처리 파트
================================================================================

본 문서는 1주차 병렬 개발을 위해 생성한 공통 데이터셋 및 전처리 통계 파일에 대한
규격 및 사용 가이드입니다.


[1] 생성된 아웃풋 파일 목록
--------------------------------------------------------------------------------
1. mini_samples.npz
   - 용도: 모델 생성, 입출력 형상(shape) 검증, 추론/평가 코드 단위 테스트용
   - 구성: 학습용 60개(클래스당 10개), 검증용 30개(클래스당 5개)
   - 대상: 전원 (② 모델 학습, ③ 비교 모델, ④ 평가, ⑤ 추론)

2. preprocess.npz
   - 용도: 정식 학습 및 추론 시 동일하게 적용할 표준화 통계 및 PCA 축
   - 구성: 시계열 채널별 평균/표준편차, 561 특징 열별 평균/표준편차, PCA 100차원 축
   - 대상: ② 모델 학습, ⑤ 추론 시스템

3. processed_data.npz
   - 용도: 피험자 기준 분할이 완료된 전체 데이터셋 압축 캐시 (텍스트 로딩 불필요)
   - 구성: fit(5,577개), val(1,775개), test(2,947개) 전체 배열
   - 대상: ② 모델 학습, ④ 평가

4. data_summary.txt
   - 용도: 분할 표본 수, 클래스별 분포, 텐서 차원 일치 여부 검증 리포트
   - 대상: 전원 (문서 및 발표자료 근거)


[2] 공통 데이터 규약 (계획서 18절 기준)
--------------------------------------------------------------------------------
* 시계열 텐서 (Xs)
  - 타입: float32
  - 형상: (배치크기 B, 128시점, 9채널)
  - 9개 채널 순서:
    1~3: body_acc (x, y, z)
    4~6: body_gyro (x, y, z)
    7~9: total_acc (x, y, z)

* 가공 특징 (Xf)
  - 타입: float32
  - 형상: (배치크기 B, 561)

* 정답 라벨 (y)
  - 타입: int64 (0 ~ 5)
  - 클래스 매핑:
    0: 걷기 (WALKING)
    1: 계단 오르기 (WALKING_UPSTAIRS)
    2: 계단 내려가기 (WALKING_DOWNSTAIRS)
    3: 앉기 (SITTING)
    4: 서기 (STANDING)
    5: 눕기 (LAYING)

* 표본 식별자 (sample_id)
  - 형식: "train:000079", "test:000000" 등 원본 분할과 원 행 번호 결합 문자열

* 피험자 기준 분할 표본 수 (중복/누수 방지)
  - 학습 세트 (fit) : 16명 (총 5,577개 구간) -> 모델 가중치 학습에만 사용
  - 검증 세트 (val) :  5명 (총 1,775개 구간) -> 피험자 ID [1, 6, 14, 21, 23] 고정
  - 시험 세트 (test):  9명 (총 2,947개 구간) -> 9주차 최종 평가 전까지 사용 금지


[3] 팀원별 사용 예시 코드 (Python)
--------------------------------------------------------------------------------
1. mini_samples.npz 불러오기 (단위 테스트 및 입출력 점검)
   import numpy as np
   data = np.load("mini_samples.npz")
   
   fit_Xs = data["fit_Xs"]          # shape: (60, 128, 9)
   fit_Xf = data["fit_Xf"]          # shape: (60, 561)
   fit_y  = data["fit_y"]           # shape: (60,)
   fit_id = data["fit_sample_id"]   # shape: (60,)
   
   val_Xs = data["val_Xs"]          # shape: (30, 128, 9)
   val_Xf = data["val_Xf"]          # shape: (30, 561)
   val_y  = data["val_y"]           # shape: (30,)
   val_id = data["val_sample_id"]   # shape: (30,)

2. preprocess.npz 불러오기 (표준화 통계 재사용)
   import numpy as np
   stats = np.load("preprocess.npz")
   
   seq_mean = stats["seq_mean"]     # shape: (1, 1, 9)
   seq_std  = stats["seq_std"]      # shape: (1, 1, 9)
   feat_mean = stats["feat_mean"]   # shape: (1, 561)
   feat_std  = stats["feat_std"]    # shape: (1, 561)
   
   # 표준화 적용:
   # Xs_norm = (Xs - seq_mean) / seq_std
   # Xf_norm = (Xf - feat_mean) / feat_std

3. processed_data.npz 불러오기 (전체 모델 정식 학습용)
   import numpy as np
   full = np.load("processed_data.npz")
   
   # 학습 세트 (fit 16명)
   Xs_fit, Xf_fit, y_fit = full["fit_Xs"], full["fit_Xf"], full["fit_y"]
   # 검증 세트 (val 5명)
   Xs_val, Xf_val, y_val = full["val_Xs"], full["val_Xf"], full["val_y"]
   # 시험 세트 (test 9명)
   Xs_test, Xf_test, y_test = full["test_Xs"], full["test_Xf"], full["test_y"]


[4] 데이터 누수 방지 주의사항
--------------------------------------------------------------------------------
1. 피험자 ID(subject)는 데이터 분할과 오류 분석에만 쓰며, 모델 입력(특징)에
   절대 포함하지 않습니다.
2. 검증 세트(val)와 시험 세트(test)의 표본은 표준화 통계(평균/표준편차) 산출이나
   PCA 축 계산에 절대 포함해서는 안 됩니다. (오직 fit 데이터로만 추정 완료됨)
3. 시험 세트(test)를 보며 모델 구조나 하이퍼파라미터를 선택하지 않습니다.
================================================================================
```[cite: 1]