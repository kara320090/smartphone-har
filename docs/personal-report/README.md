# 이봉헌 개인 보고서와 발표자료

담당 ② 기준모델 구현·학습 및 팀장 영역에서 실제 구현·실행한 내용을 정리했습니다.

| 자료 | 구성 | 파일 |
|---|---|---|
| 상세 보고서 | 17쪽, 데이터·모델·역전파·실험·해석·검증·팀 인계·참고자료 | [Word](bongheon-report.docx) · [MD 원문](REPORT_KO.md) |
| 발표자료 | 본문 12장 + 보충 5장, 발표자 노트 포함 | [PowerPoint](bongheon-presentation.pptx) |
| 발표 원고 | 장별 설명, 출처, 4분 축약 경로, 예상 질문 | [원고 MD](SPEAKER_NOTES_KO.md) |
| 산출물 검사 | 페이지·슬라이드·편집 가능 개체 수 및 파일 해시 | [검사 기록](artifact_validation.json) |

PPT의 **1 → 3 → 4 → 6 → 7 → 12번** 순서가 4분 축약 경로입니다. 각 슬라이드 노트의 시간은 연습용 목표이며 실제 발표 시간은 말하는 속도에 따라 달라집니다.

보고서는 Word에서 17쪽으로 렌더링해 확인했습니다. PPT는 PowerPoint에서 17장 전체를 확인했으며 표 7개와 그래프 3개는 수정 가능한 개체입니다. 그래프 데이터는 호환성을 위해 소수점 6자리로 저장하고, 원래 정밀도의 실험 값은 ../../reports/experiment_summary.csv에 유지했습니다.

성능은 **seed 2026의 검증 결과**입니다. 공식 시험 성능, 반복 seed 결과, 다른 팀원 모듈의 실제 통합과 시연은 후속 작업입니다. 모델·코드가 필요한 경우 [실행 안내](https://github.com/kara320090/smartphone-har/blob/main/docs/RUN_GUIDE_KO.md)와 [학습 모델 v1 릴리스](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-part2-v1)를 이용하세요.
