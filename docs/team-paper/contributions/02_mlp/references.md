# MLP 원고 참고문헌

아래 자료는 2026-10-12에 원문 또는 공식 문서를 확인했다. 라이브러리 웹 문서의 현재 버전과 실제 실행 버전은 다를 수 있으며, 실행은 고정 requirements 및 metadata의 버전을 따른다.

1. Reyes-Ortiz J., Anguita D., Ghio A., Oneto L., Parra X. **Human Activity Recognition Using Smartphones [Dataset]**. UCI Machine Learning Repository, 2013. DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K). [공식 데이터 설명](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). 사용: 센서 구간·6클래스·공식 데이터 출처. 라이선스 CC BY 4.0.
2. Anguita D., Ghio A., Oneto L., Parra X., Reyes-Ortiz J. L. **A Public Domain Dataset for Human Activity Recognition Using Smartphones**. ESANN, 2013, pp. 437–442. [학회 원문](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf). 사용: 데이터 생성·센서 구간의 배경. 본 MLP 점수와 원문 SVM 점수는 입력·선택·평가 조건이 다르므로 직접 순위 비교하지 않음.
3. Keras 3. **Flatten layer**. [공식 API](https://keras.io/api/layers/reshaping_layers/flatten/). 사용: 배치 축을 유지한 입력 펼치기.
4. Keras 3. **Dense layer**. [공식 API](https://keras.io/api/layers/core_layers/dense/). 사용: affine 변환·활성화·기본 Glorot uniform/zero bias 및 선형 logits.
5. Kingma D. P., Ba J. **Adam: A Method for Stochastic Optimization**. ICLR, 2015; arXiv:1412.6980 (최초 등록 2014). [저자 원문](https://arxiv.org/abs/1412.6980). 사용: 실제 사용한 Adam optimizer의 출처.
6. Keras 3. **EarlyStopping**. [공식 API](https://keras.io/api/callbacks/early_stopping/). 사용: val_loss·patience·최선 가중치 복원.
7. scikit-learn. **f1_score**. [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). 사용: 모든 여섯 라벨을 명시한 macro F1, zero_division=0.

구현 출처: [smartphone-har](https://github.com/kara320090/smartphone-har), 기반 커밋 `5f760c3d681505f21b3a34b03a4f909f13c13e81`의 `src/models/mlp.py`·공통 trainer 및 처리 어댑터. 원본 통계는 [bongheon-lstm-v1](https://github.com/kara320090/smartphone-har/releases/tag/bongheon-lstm-v1) 공개 릴리스에서 복원했고 해시를 대조했다. 이번 M01의 새 코드·결과와 모델 인계는 [yoonseop-mlp-v1](https://github.com/kara320090/smartphone-har/releases/tag/yoonseop-mlp-v1)을 사용한다.
