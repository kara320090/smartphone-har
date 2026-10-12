# CNN C01 TensorFlow 검증 완료 모델

세 seed의 `.keras` 파일과 각각의 동일 검증 예측·전처리·metadata를 보관합니다. `manifest.json`의 SHA-256을 확인하세요. 공식 test는 사용하지 않았습니다.
동일 입력이 필요하면 프로젝트 루트에서 다음 명령으로 재검증할 수 있습니다.

```bash
python -m src.verify_bundle --processed-data data/processed_data.npz --preprocess-stats data/preprocess.npz --run-directory artifacts/cnn_tensorflow/C01_seed2026 --report reports/cnn/reverify_seed2026.json
```
