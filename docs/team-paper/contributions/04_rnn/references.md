# RNN 참고문헌과 사용 위치

[1] Reyes-Ortiz 외. Human Activity Recognition Using Smartphones. UCI Machine Learning Repository, 2013. DOI: 10.24432/C54S4K. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones — 데이터 출처와 128시점 센서 구간 설명.

[2] Keras. SimpleRNN layer. 공식 API 문서, 확인 2026-10-12. https://keras.io/api/layers/recurrent_layers/simple_rnn/ — tanh 순환층, 마지막 출력, 독립 구간 상태 정책. 문서의 게시 연도는 확인되지 않았다.

[3] Keras. EarlyStopping. 공식 API 문서, 확인 2026-10-12. https://keras.io/api/callbacks/early_stopping/ — validation loss 기준 종료와 최선 가중치 복원. 문서의 게시 연도는 확인되지 않았다.

구현 근거: 제공 main의 src/train.py, src/adapters/processed_npz.py와 이번 src/models/rnn.py. 최종 공동 원고에서는 참고문헌 중복을 제거하고 인용 번호를 통일한다.
