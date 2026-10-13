# 동일 피험자 분할에서 스마트폰 행동 인식 신경망 네 모델의 개발 검증 비교

**양식 편집본:** 아래는 초기 통합 MD이며, 사용자 첨부 대학생 양식에 맞춘 최신 [HWP·같은 본문 MD·2쪽 검증](../../dcs-team-paper-20261013/README.md)을 우선 사용한다.

**내부 통합 초안 · 2026년 10월 13일.** 저자·순서·소속·지도교수·이메일: 확인 필요. RNN 대표·최종 평가 규약은 팀 확인 후 확정한다. 학회에 제출한 파일이 아니다. 지면 검증 전 MD 원고이며 원고 전체를 학생 HWP 양식의 1~2쪽으로 편집해야 한다.

## 1. 서론

스마트폰 센서 기반 행동 인식에서 입력 구간을 처리하는 신경망 구성에 따른 성능을 살펴보았다. 동일한 데이터·사람 분할·학습 조건에서 MLP, 1D CNN, SimpleRNN, LSTM의 개발 검증 성능과 모델 규모를 비교하였다. 이 비교는 각 담당자가 고정한 구성의 비교이며 파라미터 수나 정규화를 동일하게 맞춘 구조 효과의 통제 실험은 아니다.

## 2. 데이터와 공통 실험

UCI HAR[1]의 전처리된 128시점·9채널 구간과 여섯 행동 라벨을 사용하였다. 채널은 body acceleration, body gyroscope, total acceleration의 xyz 순이다. 학습 16명·5,577구간과 검증 5명·1,775구간의 사람이 겹치지 않도록 분리하였다. 저장된 fit 통계로 표준화를 한 번 적용했으며 PCA는 사용하지 않았다. 센서의 완전한 원시 신호를 사용한 것은 아니다. **데이터 담당자는 원 전처리·분할 설명과 출처를 최종 확인한다.**

Adam 학습률 0.001, batch 64, 최대 40 epoch, validation loss 기준 patience 6으로 학습하고 최저 validation loss의 가중치를 복원하였다. 각 모델은 seed 2026·2027·2028에서 반복하였다. 검증 데이터는 epoch 선택에 사용한 개발 자료이며 공식 test는 평가하지 않았다. 학습은 각 담당자의 CPU 환경에서 수행하여 개인 PC의 시간을 속도 비교에 사용하지 않았다.

## 3. 모델

MLP는 입력을 1,152차원으로 펼쳐 Dense 128·64 ReLU와 6 logits로 연결하였다. CNN은 시간축 Conv1D의 64(k5)·128(k3)·128(k3) 필터를 사용하며 첫 두 합성곱 뒤 Pool 2, 마지막 합성곱 뒤 GAP와 Dense 64 ReLU, Dropout 0.3을 적용하였다. SimpleRNN은 tanh 은닉 32에서 마지막 상태를 6 logits로 변환하였다. LSTM[2,3]은 은닉 64의 마지막 상태를 Dense 32 ReLU와 6 logits에 연결하였다. RNN·LSTM의 각 구간 시작 상태는 0이며 다른 기본 모델의 Dropout은 0이다.

![그림 1. 실제 네 기본 모델 구성](figures/model_architectures.png)

## 4. 개발 검증 결과와 한계

표 1. 동일 사람 분할에서 세 seed의 개발 validation 평균 ± 표본 SD.

| 모델 | 조건 | 파라미터 | Dropout | Accuracy 평균 ± SD | Macro F1 평균 ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP | M01 | 156,230 | 0 | 0.8667 ± 0.0117 | 0.8646 ± 0.0125 |
| CNN | C01 | 85,574 | 0.3 | 0.8888 ± 0.0116 | 0.8889 ± 0.0127 |
| RNN | R01 | 1,542 | 0 | 0.7356 ± 0.0497 | 0.7191 ± 0.0586 |
| LSTM | L01 | 21,222 | 0 | 0.8832 ± 0.0194 | 0.8813 ± 0.0195 |

CNN의 Macro F1 평균은 0.8889, LSTM은 0.8813로 관찰되었다. 파라미터는 각각 85,574개와 21,222개이다. 저장 예측을 동일한 6클래스 계산으로 재집계하여 게시 수치와의 일치를 확인하였다. 표준편차는 고정 분할의 학습 seed 변동이며 신뢰구간이 아니다. 작은 seed 수와 단일 사람 분할, validation의 모델 선택 사용으로 일반화 우월성을 단정할 수 없다.

RNN의 개인 튜닝 선택 조건 R02(은닉 64)는 Macro F1 0.7715 ± 0.0675였다. 이 결과는 R01 기준 비교와 구분하였다. 최종 표에 R02를 사용할 경우 각 모델의 탐색 예산과 선택 규칙을 함께 명시한다. 단순히 가장 높은 조건을 모델별로 골라 공정한 동일 예산 비교라고 서술하지 않는다.

## 5. 결론

동일한 센서 입력·사람 분할에서 네 고정 모델 구성의 반복 개발 검증 결과를 취합하였다. CNN과 LSTM의 비교적 높은 평균 및 파라미터 규모 차이를 관찰했으나, 구성·정규화 차이와 모델 선택에 사용한 검증 지표라는 한계가 있다. 최종 대표 조건과 test 평가 규약, 저자 정보를 확인한 뒤 제출본을 확정한다.

## 참고문헌

[1] UCI Machine Learning Repository, “Human Activity Recognition Using Smartphones,” 2013. DOI: 10.24432/C54S4K. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

[2] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, 9(8), 1735–1780, 1997. https://www.bioinf.jku.at/publications/older/2604.pdf

[3] F. A. Gers, J. Schmidhuber, and F. Cummins, “Learning to Forget: Continual Prediction with LSTM,” Neural Computation, 12(10), 2451–2471, 2000. https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf

**취합 근거:** [15회 예측 재계산·범위와 남은 확인 사항](README.md). 이 문서의 표는 `reports/team_comparison/baseline_summary.csv`에서 생성했다. 다른 담당자의 상세 보고서를 이어 붙이는 방식 대신 공통 조건을 한 번 설명했다. 실제 HWP 편집 시 그림 번호·참고문헌 번호와 전체 1~2쪽을 검증한다.
