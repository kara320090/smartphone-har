# SimpleRNN 구조와 상태 정책

| 층 | R01 출력 / 파라미터 | R02·R03 출력 / 파라미터 |
|---|---|---|
| 입력 | B×128×9 / 0 | B×128×9 / 0 |
| SimpleRNN tanh | B×32 / 1,344 | B×64 / 4,736 |
| 출력 Dropout | B×32, 비율 0 / 0 | B×64, R02=0·R03=0.3 / 0 |
| Dense logits | B×6 / 198 | B×6 / 390 |
| 합계 | 1,542 | 5,126 |

은닉층 1개, stateful=False, return_sequences=False. 매 구간의 h0=0이며 마지막 h128을 분류한다. 순환 내부·입력 Dropout은 0, clipping은 사용하지 않는다. R03은 같은 seed의 R02 학습 전 가중치를 복사했다.

공동 기준 비교는 R01이다. 개인 보조 튜닝에서 선택된 R02와 구분한다.
