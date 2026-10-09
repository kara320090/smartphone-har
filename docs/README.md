# 문서 목차

처음 보는 팀원은 **최신 공동 논문 안내 → 실행 안내 → 팀 인계** 순서로 읽는다. 최신 배정은 데이터 정회서·MLP 한윤섭·CNN 최승빈·RNN 유재윤·LSTM 이봉헌이며 전처리는 완료되었다. 기존 이봉헌 MLP 보고서는 이전 수행 기록이다. 결과를 검토할 때는 실험 결과와 조건별 해석을 함께 확인한다.

| 문서 | 용도 |
|---|---|
| [공식 대학생 HWP 취합 초안](dcs-student-paper/README.md) | 사용자 제공 양식의 2쪽 한글 파일·수정용 MD·제출 확인표·배치와 수치 검증 |
| [현재 LSTM 완료 자료](lstm-report/README.md) | 9회 실험, 22쪽 Word·PDF·MD, 20장 PPT, 질의응답 32개, 재현·추론 명령 |
| [LSTM 실제 논문 기여 원고](team-paper/contributions/05_lstm_bongheon.md) | 방법·조건·수치·한계, 구조표와 세 seed 결과표 |
| [팀 공동 논문 역할과 작성 안내](team-paper/README.md) | 5인 역할별 작업·결과물·원고 양식, 공동 실험 규칙, 1편 취합 절차와 내부 일정 제안 |
| [이전 MLP 심화 보고서와 PPT](process-report/README.md) | 실제 결함 수정, 24회 학습, 21쪽 Word, 20장 PPT, 질의응답 34개 |
| [이봉헌 담당 업무 정리](BONGHEON_WORK_SUMMARY.md) | 완료한 일·실제 결과·담당 파일·남은 공동 작업 |
| [개인 보고서와 PPT](personal-report/README.md) | 17쪽 Word 보고서, 17장 PPT, MD 원문, 장별 발표 원고 |
| [시작 안내](../START_HERE_KO.md) | 환경 준비, 실행, 저장 모델 사용 |
| [설치와 실행 상세 안내](RUN_GUIDE_KO.md) | 명령어, 데이터 계약, 설정, 저장 형식 |
| [실험 결과](../reports/RESULTS_KO.md) | 7개 실험 점수, 학습 곡선, 한계 |
| [조건별 가설과 해석](EXPERIMENT_NOTES_KO.md) | 바꾼 조건과 유지 조건, 관측, 가능한 설명 |
| [검증 기록 요약](../reports/VERIFICATION_KO.md) | 테스트·역전파·TFRecord·재로딩 근거 |
| [팀 모듈 인계](HANDOFF_KO.md) | 데이터·모델·평가·추론 담당자와 맞출 계약 |
| [발표 원고](PRESENTATION_KO.md) | 약 4분 설명과 예상 질문 |
| [팀장 작업표](TEAM_TASKS_KO.md) | 연결 순서와 진행 확인 |

## 실행된 노트북

1. [미분과 역전파 검증](../notebooks/01_backprop_validation.ipynb)
2. [공통 학습과 모델 저장](../notebooks/02_training_walkthrough.ipynb)
3. [학습 조건 비교](../notebooks/03_condition_comparison.ipynb)

## 근거 파일

[실험 요약 CSV](../reports/experiment_summary.csv) · [강의 근거 목록](../evidence_index.csv) · [전달물 검사](../reports/delivery_audit.json)
