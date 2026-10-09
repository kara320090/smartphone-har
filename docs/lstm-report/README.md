# 이봉헌 LSTM 완료 자료

정식 9회 학습, 39개 코드 테스트, 작은 LSTM의 174회 수치미분, 전체 validation 재로딩 검증을 완료했다. 기본 L01의 검증 Macro F1은 **0.8813 ± 0.0195**다. 공식 test는 미평가다.

- [22쪽 Word 보고서](bongheon-lstm-report.docx) · [PDF](bongheon-lstm-report.pdf) · [Markdown 원문](REPORT_KO.md)
- [20장 PPT: 본문 15 + 부록 5](bongheon-lstm-presentation.pptx) · [발표 원고](SPEAKER_NOTES_KO.md)
- [팀 논문용 LSTM 원고와 표](LSTM_CONTRIBUTION_KO.md)
- [질의응답 32개](DEFENSE_QA_KO.md) · [실제 과정 기록](PROCESS_LOG_KO.md)
- [실행·검증·추론 명령](RUN_GUIDE_LSTM_KO.md) · [팀 입력·결과 규격](HANDOFF_LSTM_KO.md)
- [문서 검증과 수정 기록](ARTIFACT_QA_KO.md)

[결과·검증 원자료](../../reports/lstm/) · [실행 계획과 변경 기록](../LSTM_EXECUTION_PLAN_KO.md) · [학습 모델과 상세 근거 Release](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-lstm-v1)

공동 비교에는 L01 세 seed를 제공한다. L02·L03은 보조 조건이며 결과 후 대표 모델을 교체하지 않았다. 실제 결함·코드 연결 차이·의도적 오류 주입을 구분했고, 측정하지 않은 팀원 모델의 점수나 test 점수를 작성하지 않았다. 발표 전에 본인이 수식과 코드, 실제 결과를 확인하고 공동 논문 원고는 팀원 결과와 함께 최종 검토한다.
