# 검증 기록 요약

Windows CPU, Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1에서 실행했다. 문서 정리 과정에서는 학습 코드·설정·모델·측정 수치를 바꾸지 않았다.

| 확인 항목 | 결과 | 원시 근거 |
|---|---|---|
| 자동 테스트 | 8개 통과, 실패 0개 | [tests.xml](tests.xml) |
| 노트북 | 3개 모두 실행, 코드 셀 총 13개 | [notebook_execution.json](notebook_execution.json) |
| 미분 검증 | 17개 파라미터 × 3개 h, 총 51개 비교 통과 | [gradient_checks.json](math/gradient_checks.json) · [세부 CSV](math/gradient_details.csv) |
| NumPy와 자동미분 | 최대 절대 오차 약 5.55×10⁻¹⁷ | [수치 결과](math/gradient_checks.json) |
| TFRecord | 학습 200개 표본의 값·shape·정답·사람·ID 정확히 일치 | [tfrecord_check.json](tfrecord_check.json) |
| 저장물 재현 | 모델 7개 × 검증 1,775개, 새 프로세스에서 확률·라벨·지표 재현 | [fresh_process_verification.json](fresh_process_verification.json) |
| 전달 파일 | 코드 해시·모델 해시·전처리 해시·실험 상태 확인 | [delivery_audit.json](delivery_audit.json) |
| 데이터 출처 | 공식 ZIP SHA256이 계획서와 일치 | [data_provenance.json](data_provenance.json) |

## 자동 테스트의 범위

학습 데이터만 사용한 전처리, PCA 저장·복원, 시계열 채널별 통계, 사람·표본 ID 중복 거부, 파라미터 수, 초기화 재현, 6개 클래스 macro F1, 안정적인 softmax, 역전파, 학습 저장·재로딩, 덮어쓰기 방지, 외부 분류·재구성 모델 연결을 확인했다.

## 이번 검증에 포함하지 않은 항목

공식 시험 데이터 평가, 전체 팀 모델의 seed 반복, 팀원의 실제 코드 연결, 다른 사람의 PC·Colab 실행, 모바일 환경, 완성 시연과 지연시간 측정은 포함하지 않았다. 각 담당자의 후속 작업과 구분해 기록한다.
