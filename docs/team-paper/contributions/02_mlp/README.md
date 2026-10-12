# 한윤섭 MLP 완료 자료

2026-10-12 · MLP 작업 완료 · GitHub 저장소·Release 인계<br>
M01: Flatten → Dense 128/ReLU → Dense 64/ReLU → 6 logits, 파라미터 156,230개.

**validation Macro F1 0.8646 ± 0.0125, Accuracy 평균 86.67%, seed 표본 SD 1.17%p.** 동일한 사람 분할의 3seed 변동이다. 공식 test는 미평가다.

[**전체 ZIP 다운로드**](https://github.com/kara320090/smartphone-har/releases/download/yoonseop-mlp-v1/yoonseop-mlp-v1-full.zip) · [Release와 SHA-256 목록](https://github.com/kara320090/smartphone-har/releases/tag/yoonseop-mlp-v1)

ZIP에는 코드·문서·그림·전처리 데이터·원본 통계·저장 모델 3개·전체 예측·검증 근거를 함께 담았다. Git clone에는 가상환경·데이터·모델이 포함되지 않으므로 실행 전에 ZIP을 내려받거나 해당 파일을 준비한다. seed별 설정·학습 곡선 원자료·예측·지표·재로딩 기록은 [run_evidence](../../../../reports/mlp_yoonseop/run_evidence/)에서 바로 읽을 수 있다.

| 바로 볼 자료 | 파일 |
|---|---|
| 개인 논문 원고·결과·오류·한계 | [contribution.md](contribution.md) |
| 공동 논문용 짧은 문단·결과 한 행 | [paper_insert.md](paper_insert.md) |
| 실행·재로딩·추론 방법 | [RUN_GUIDE_KO.md](RUN_GUIDE_KO.md) |
| 학습 전에 고정한 조건 | [training_protocol.md](training_protocol.md) |
| 주장별 원자료 | [evidence_index.md](evidence_index.md) |
| 확인한 참고문헌·기존 코드 출처 | [references.md](references.md) |
| seed별 지표·데이터·코드 버전 | [runs.csv](../../../../reports/mlp_yoonseop/runs.csv) |
| 그래프·분석·검사·고정 소스 | [reports/mlp_yoonseop](../../../../reports/mlp_yoonseop/) |

정식 모델은 `runs/mlp_yoonseop/M01_seed2026`, `M01_seed2027`, `M01_seed2028`에 있다. 48개 코드 검사, 소량 연결 검사, 모델별 1,775개 validation 전체의 별도 프로세스 재로딩이 통과했다. 잘못된 7종 추론 입력을 거부했고 예측 파일에서 지표를 재계산하였다.

새 작업은 기존 E03 구조·공통 trainer를 사용한 한윤섭 담당 M01 실행과 원고·인계다. 이전 이봉헌 MLP 결과를 이번 3seed 결과에 합치지 않았다. 데이터·통계는 기존 LSTM 자료와 같은 해시이며 직접 fit하지 않았다. 초기 임시 폴더 문제·중단 run은 보존하고 정식 결과에서 제외했다.

팀 최종 test 시점·공동 비교·저자/소속·학회 투고는 팀 후속 범위다. 이번 자료를 읽은 팀원이나 작성자의 최종 검토 완료를 대신 표시하지 않았다. 코드·원고·지표·seed별 증빙은 저장소에, 모델·데이터를 포함한 전체 묶음은 `yoonseop-mlp-v1` Release에 제공한다. 학습 때 고정한 코드와 통계·예측은 보존하며 배포 준비 과정에서 모델이나 실측 점수를 바꾸지 않았다.
