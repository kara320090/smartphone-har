# v7 연속 2단 지면과 그림 읽는 순서

공식 대학생 양식의 용지·여백·휴먼명조 본문 10pt·160% 줄 간격을 유지하고 전체 본문을 연속 2단으로 배치했다. 제목·저자·소속·지도교수와 기존 결과는 유지했다. 그림·표 주변에는 원양식의 9pt 가운데 문단 형식으로 빈 문단을 남겼다.

| 개체 | 내용 | 읽는 기준 |
| --- | --- | --- |
| 그림 1 | 네 고정 구성의 분류 구조 | MLP·CNN·RNN·LSTM의 실제 층과 구간별 상태 초기화. 측정된 활성값 그림이 아니다. |
| 표 1 | 파라미터 수와 Macro F1 평균±SD | 전체 검증 5명·1,775구간의 기준 결과. SD는 세 seed의 표본 SD다. |
| 그림 2(a) | 다섯 피험자×네 구성의 Macro F1 평균 | 각 사람 안에서 계산한 6클래스 F1의 seed 평균. n은 매 seed의 구간 수이고 색 범위는 0.5-1.0이다. |
| 그림 2(b) | 전체·다섯 제외 진단의 CNN-LSTM 평균 차이 | 양수는 CNN, 음수는 LSTM 평균이 높다. 재학습하지 않았고 대표 결과에는 전체 다섯 명을 유지한다. |

두 본문 그림은 각각 80mm 폭의 600dpi PNG이며 SVG도 제공한다. 그래프 라벨은 소수 세 자리, 표·본문은 네 자리다. 전체 +0.008은 원래 +0.0075933765, 14 제외 -0.004는 원래 -0.0044281056이다. 정확한 값·단위·해시는 `VISUAL_FIGURE_DATA.json`에 있다.

```powershell
python tools/make_research_figure.py --paper-layout compact --output docs/dcs-team-paper-research-v7
```

이 명령은 그림과 표시 데이터 JSON을 생성한다. HWP를 자동 갱신하거나 학습을 수행하지 않는다. 성능·규모 산점도가 포함된 v6의 큰 그림 `figures/research_summary.*`는 보충 자료이며 v7 본문 HWP에는 들어가지 않는다. 해당 그림의 데이터는 `SUPPLEMENTARY_V6_FIGURE_DATA.json`에 보존했다.

비교표는 `reports/team_comparison/baseline_summary.csv`, 피험자 값은 `reports/team_error_analysis/subject_summary.csv`, 구성 진단은 `reports/subject_composition/paired_seed_gaps.csv`다. 72개 진단 지표를 재계산해 scikit-learn과 대조한 기존 근거를 유지했다. 원고를 한글로 저장·재로딩하고 두 쪽 PDF 전체를 렌더링해 본문·표·축·숫자·캡션을 확인했다.
