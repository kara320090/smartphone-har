# 저장한 LSTM 모델의 수업 시연

이번 시연은 준비된 validation 구간 다섯 개를 분류하는 오프라인 실행이다. 공식 test, 스마트폰 실시간 수집, 모델의 모든 행동에 대한 성능 검증과 구분한다. 다섯 예시는 val 배열의 첫 다섯 행이며 정답이 모두 STANDING이다. 다른 활동의 결과는 전체 혼동행렬에서 설명한다. 이 작은 시연의 정답 수를 모델 전체 성능으로 사용하지 않는다.

## 준비

저장소의 requirements.txt로 설치한 Python 환경, 원래 해시의 processed_data.npz, 기존 LSTM Release의 L01_seed2026 실행 폴더가 필요하다. ZIP을 저장소 루트에 풀면 runs/lstm/L01_seed2026 경로를 확인할 수 있다. 모델 파일과 대응 전처리·metadata를 함께 유지한다. [전체 실행 안내](../lstm-report/RUN_GUIDE_LSTM_KO.md)를 참고한다.

저장소 루트에서 실행한다. PowerShell의 활성화된 Python 환경을 사용하고 새 출력 파일명을 지정한다.

```powershell
python -m analysis.lstm_predict --run-directory runs/lstm/L01_seed2026 --processed-data "C:/Users/soma/Desktop/processed_data.npz" --count 5 --output reports/class_demo_new.json
Get-Content -LiteralPath reports/class_demo_new.json
```

기존 출력이 있으면 덮어쓰기를 막으므로 class_demo_new_2.json처럼 새 파일명을 사용한다. CLI는 저장 metadata의 데이터 SHA256을 대조하고 val_Xs·val_sample_id만 읽으며 test 배열에 접근하지 않는다. 정답은 추론 입력에 포함하지 않는다. 아래 정답 열은 이미 보관한 validation_predictions.csv에서 연결해 시연 후 확인한 값이다.

## 실제 실행 결과

| sample_id | 사람 | 정답 | 예측 | 최대 softmax 값 |
|---|---:|---|---|---:|
| train:000000 | 1 | STANDING | STANDING | 0.8903 |
| train:000001 | 1 | STANDING | STANDING | 0.8831 |
| train:000002 | 1 | STANDING | STANDING | 0.8991 |
| train:000003 | 1 | STANDING | STANDING | 0.8983 |
| train:000004 | 1 | STANDING | STANDING | 0.8898 |

2026년 10월 9일 새 프로세스에서 실행했다. 기존 저장 예측과 클래스는 모두 일치했으며 확률 최대 절대 차이는 6.605e-08였다. softmax 값은 보정된 신뢰도나 정확할 확률의 보증이 아니다.

발표 설명: 저장 모델과 전처리를 함께 불러오고, 128×9 입력을 한 번 표준화하여 분류했습니다. 입력 구간의 출처와 ID를 보존해 저장된 결과와 대조했습니다.
