# RNN 자료 위치와 검증 기록

- [PR #3](https://github.com/kara320090/smartphone-har/pull/3): RNN 코드·설정·실험 원자료·노트북·문서
- [Release](https://github.com/kara320090/smartphone-har/releases/tag/yu-jaeyun-rnn-v1): 저장 모델·초기 가중치·입력 데이터·전처리 통계·재현 묶음
- 브랜치: feat/yu-jaeyun-rnn

정식 실험은 R01·R02·R03 × seed 2026·2027·2028의 9회다. 기준 R01의 검증 Macro F1은 0.7191 ± 0.0586, 튜닝 선택 R02는 0.7715 ± 0.0675다. 공식 test는 미평가다.

통합 테스트 68개, RNN 전용 묶음 테스트 11개, 세 조건의 연결 점검이 통과했다. 저장 모델 9개를 검증 1,775개 전체에서 새 프로세스로 재로딩했고, 노트북의 코드 5개 셀이 정상 실행됐다.

모델과 입력 데이터는 Release에서 받는다. 실행 기록과 게시 소스의 관계는 [소스 이력 설명](../../reports/rnn/SOURCE_PROVENANCE_KO.md), Release 파일별 해시는 RNN_RELEASE_MANIFEST.json과 SHA256SUMS.txt에 있다.
