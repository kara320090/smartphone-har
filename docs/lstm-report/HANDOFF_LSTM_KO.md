# 팀에 넘기는 LSTM 입력·결과 규격

담당: 이봉헌. 데이터 담당 정회서의 캐시와 통계를 사용했다. 공동 비교 기준은 L01 세 seed다.

| 항목 | 규격 |
|---|---|
| 입력 | 표준화 전 float32 (N,128,9), 팀 저장 통계를 한 번 적용 |
| 채널 순서 | body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z |
| 라벨 | 0 WALKING, 1 WALKING_UPSTAIRS, 2 WALKING_DOWNSTAIRS, 3 SITTING, 4 STANDING, 5 LAYING |
| 분할 | team_s0_val_1_6_14_21_23; fit 5577, val 1775 |
| ID | train:NNNNNN, 공식 train의 0 기반 행 번호 |
| seed | 2026, 2027, 2028 |
| 학습 | Adam .001, batch 64, 최대 40, val_loss patience 6, 최선 가중치 복원 |
| 기준 성능 | validation Macro F1 0.8813 ± 0.0195 |
| 결과 범위 | 고정 사람 분할의 개발 validation, 공식 test 미평가 |

## 결과 파일

각 실행의 validation_predictions.csv는 sample_id, subject, true_label, predicted_label, logit_0부터 logit_5, probability_0부터 probability_5를 담는다. 확률의 열은 위 클래스 순서이며 argmax가 predicted_label이다. ID의 앞 0을 지우거나 기존 1 기반 ID와 바로 조인하지 않는다. predictions를 동일 ID 순서로 대조한 뒤 공통 평가 코드로 Accuracy와 6클래스 Macro F1을 계산한다.

reports/lstm/에는 각 실행·클래스·사람의 지표와 support, seed 평균·표본 SD, 같은 seed의 조건 차이, 혼동행렬, 오분류 ID와 신호 통계가 있다. 전체 모델·초기 가중치·로그·예측은 [Release](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-lstm-v1)의 ZIP 안에 있다. ZIP은 저장소 루트에서 재현할 수 있는 소스와 frozen-source 폴더의 학습 당시 소스를 함께 담는다.

## 네 모델을 취합할 때 확인

1. 데이터·통계 SHA256, 사람 분할, ID와 정답 순서를 대조한다.
2. MLP의 561특징 사용과 시계열 모델의 128×9 사용처럼 입력 표현이 다르면 논문에 밝힌다.
3. 모델별 설정·파라미터·seed·epoch 선택·탐색 예산을 표시한다.
4. 예측 파일에서 지표를 재계산하고 같은 장치·측정 범위의 시간만 비교한다.
5. L01 세 seed를 기준 비교표에 넣고 L02·L03은 보조 분석으로 구분한다.
6. 모든 담당자의 실제 결과가 모인 뒤 공동 논문을 취합하고 공식 test 평가 절차를 합의한다.

입력 해시와 실제 사람 목록은 reports/lstm/dataset_audit.json에 있다. 원본 대조는 analysis/lstm_data_audit.py로 재현한다. 이 인계에 다른 담당자 모델의 새 실험 결과는 포함하지 않았다.
