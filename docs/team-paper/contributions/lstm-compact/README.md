# 이봉헌 LSTM 공동 논문 삽입 자료

공동 논문 전체가 1~2쪽이라는 기준에 맞춰 **이봉헌의 LSTM 부분만 압축하고 실제 구조 그림을 제작**했다. 기존의 상세 보고서·수업 PPT·9회 실험 근거를 토대로 만들었다. 다른 팀원 결과를 대신 작성하거나 학회에 제출한 파일이 아니다.

## 바로 사용할 파일

- **LSTM_논문삽입용_본문그림.hwp**: 짧은 본문 222자와 구조 그림. 모델 문단·그림·결과 문단을 공동 원고의 각 절에 붙여 넣는 용도.
- **LSTM_COMPACT_INSERT_KO.md**: 381자 설정 포함 본문, 222자 압축 본문, 그림 제목, 통합 비교표 한 행, 참고문헌과 취합 주의점.
- **figures/lstm_l01_architecture.png**: 실제 L01 구조, 80 mm 기준 600 dpi, 1,890×675 px. 흑백 인쇄를 고려해 제작.
- **figures/lstm_l01_architecture.svg**: 같은 구조 그림의 벡터 파일. 텍스트와 배치를 수정할 수 있으며 맑은 고딕 글꼴을 사용한다.
- **LSTM_COMPARISON_ROW.csv**: 21,222개 파라미터, L01 세 seed의 개발용 validation Accuracy·Macro F1 평균과 표본 SD.
- **TEAM_PAGE_BUDGET_KO.md**: 2쪽 전체에서 역할별 설명·공통 조건·그림·비교표·참고문헌을 합치는 기준.

## 참고용 파일

`LSTM_설정포함_참고원고.hwp`에는 381자 본문과 학습 설정·구조 그림·LSTM 결과표가 있다. 최종 공동 논문에서는 이 설정을 공통 실험 절에 합치고 결과표는 네 모델 통합 표의 한 행으로 사용한다. 별도 LSTM 표를 중복해서 넣지 않는다.

`figures/lstm_auxiliary_macro_f1.png`는 이미 검증한 보고서의 L01·L02·L03 그래프를 그대로 옮긴 개인 보고서·PPT용 자료다. 점은 seed별 값, 검은 표시는 평균과 표본 SD다. 공동 비교 대표는 L01이며 보조 실험에서 더 높은 평균이 나왔다고 바꾸지 않는다.

## 수치와 범위

L01 개발용 validation Macro F1은 **0.8813 ± 0.0195**, Accuracy는 **0.8832 ± 0.0194**다. ±는 동일 사람 분할에서 세 학습 seed의 표본 SD이며 신뢰구간이 아니다. validation은 epoch 선택에도 사용했고 공식 test는 평가하지 않았다. 최종 비교 결론은 팀원 결과 취합 후 작성한다.

그림은 **128×9 → LSTM 64 → Dense 32 ReLU → Dropout 0 → Dense 6 logits**을 나타낸다. `h128`은 마지막 128번째 시점의 은닉 상태다. `h0=c0=0`은 각 입력 구간의 상태를 0으로 초기화한다는 뜻이다. 입력 표준화와 데이터 분할 설명은 공동 데이터 절에서 한 번만 설명한다.

![공동 논문에 삽입할 실제 L01 LSTM 구조](figures/lstm_l01_architecture.png)

두 HWP는 공식 학생 양식의 본문 글꼴·여백·2단을 사용한 **부분 원고**다. 한 쪽 문서의 한 단 안에 내용이 들어가며 나머지는 의도적으로 비워 두었다. 각 파일 자체를 학회 제출용 완성 논문으로 사용하지 않는다. 전체 취합 후 그림·표·참고문헌을 포함해 1~2쪽을 다시 검증한다.

## 확인 완료

두 파일을 실제 한글 HWP 형식으로 저장하고 다시 불러와 본문 보존·1쪽·그림 삽입·한 단 배치를 확인했다. 한글이 생성한 각 파일의 유일한 페이지 미리보기와 원본 그림을 검토했다. 그림 폭 약 80 mm는 본문 한 단 약 81 mm 안에 들어가고, 그림 제목은 중앙 정렬 9 pt 일반 문단이다. PNG는 저장 후에도 원본 바이트가 유지되었다. CSV 수치는 `reports/lstm/summary.csv`의 L01 결과와 일치한다. 자세한 확인 항목은 [verification.json](verification.json), 배포 파일 해시는 `SHA256SUMS.txt`에 남겼다.

[압축 자료 Release 다운로드](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-lstm-paper-fragment-v1)

## 기존 자료

- [LSTM 코드·9회 실험·상세 보고서](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-lstm-v1)
- [인공지능 공학 수업 PPT와 발표 원고](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-course-paper-v1)
- [기존 상세 HWP 취합 초안](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-dcs-student-draft-v1)

기존 2쪽 HWP는 상세한 LSTM 검토 초안이다. 팀의 최종 2쪽 원고를 취합할 때는 이번 압축 자료를 우선 사용한다.
