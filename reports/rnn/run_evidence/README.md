# 유재윤 RNN 실행 원자료

R01·R02·R03 × seed 2026·2027·2028, 총 9회의 설정·학습 곡선·예측·지표·분할·복원 기록·로그를 원본 바이트로 보존했다. 전체 저장 모델과 초기 가중치, 전처리 통계·공유 입력 데이터는 `yu-jaeyun-rnn-v1-full.zip` Release에 있다. 원본 메타데이터의 코드 해시는 당시 전체 개발 폴더에 대한 기록이며 이번 선별본의 코드 해시와 다르다. 상세 설명은 `../SOURCE_PROVENANCE_KO.md`다.

Git만 받아도 `python -m src.rnn_analysis --runs-dir reports/rnn/run_evidence --out reports/rnn-recomputed`로 검증 지표와 그림을 재계산할 수 있다. 공식 test는 미평가다.
