# 문서와 발표자료 검증 기록

Word 보고서 22쪽과 PPT 20장 전체를 렌더링하고 한글·수식·표·그래프의 겹침과 잘림을 검토했다. 보고서에 실제 오류 표본과 클래스 지표를 추가하며 초안의 표가 다음 페이지로 넘어가는 문제를 발견했다. 본문은 상위 5개, 원자료는 상위 10개를 보존하는 방식으로 배치를 조정하고 전체를 다시 렌더링했다. 표지 제목 줄바꿈도 조정했다.

문서 도구의 기본 render_docx.py는 이 Windows 환경에 번들 soffice.exe가 없어 실행되지 않았다. 별도 Word 인스턴스에서 읽기 전용 문서를 PDF로 내보내고 Poppler로 모든 페이지를 렌더링했다. 기본 렌더러가 성공했다고 기록하지 않는다. 원본 Word 문서는 python-docx로 작성했다.

PPT는 Artifact Tool의 native 표·차트를 사용했다. 최초 최종 검증 호출의 런타임 모듈 경로 누락을 수정했다. 재실행에서는 기존 검증 receipt를 덮어쓰지 않는 제약을 확인하고 초안 receipt를 보존한 뒤 최종 검증을 다시 수행했다. 학습 코드나 가중치에는 영향이 없다. loss 차트는 실제 epoch 간 직선 연결을 사용하도록 smoothing을 껐다.

Word와 PPT의 정식 9회 표를 결과 JSON과 대조했다. native 차트의 cache도 seed 지표·history 값과 대조하고 편집 가능한 차트 데이터 workbook 및 20개 발표 노트를 확인했다. delivery_verification.json과 presentation_validation.json에 해시 및 검사 범위를 기록했다. 구조 검사가 모든 응용프로그램의 동일한 표시를 보장한다는 뜻은 아니다.
