# 공동 논문 삽입용 RNN 문단

기본 순환신경망은 128시점·9채널을 한 층의 Keras SimpleRNN으로 처리한다. 은닉 유닛 32개, tanh 활성화와 stateful=False를 사용하며 매 구간의 상태를 0으로 초기화한다. 마지막 은닉 상태를 Dense 6 logits로 변환해 여섯 행동을 분류한다. 기준 모델의 파라미터 수는 1,542개다.

개발 validation의 기준 R01 세 seed Macro F1은 0.7191 ± 0.0586, Accuracy 평균은 0.7356이다. ±는 seed 표본 SD이며 공식 test는 미평가다. 통합 표에는 RNN_COMPARISON_ROW.csv 한 행을 사용한다. 개인 튜닝 승자 R02로 기준 비교 행을 교체하지 않는다. 최종 제출본은 전체 모델 결과와 담당자 검토 후 취합한다.
