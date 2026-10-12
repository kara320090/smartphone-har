"""Publish only independently verified, formal TensorFlow C01 runs to Git-tracked paths.

Usage after run_cnn_suite.py and cnn_summarize.py:
  python scripts/cnn_publish.py [--replace]

No training data or official test data is exported. Provisional Keras/Torch runs are
never accepted as formal TensorFlow results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2026, 2027, 2028]
RUN_FILES = [
    'model.keras', 'preprocess.npz', 'config.json', 'metadata.json',
    'metrics.json', 'split.json', 'history.csv', 'validation_predictions.csv',
    'initial_gradient_check.json', 'status.json', 'reload_check.json',
    'fresh_process_verification.json',
]
REPORT_FILES = [
    'cnn_summary.json', 'cnn_runs.csv', 'cnn_per_subject.csv',
    'cnn_learning_curves.png', 'cnn_confusion.png',
    *[f'C01_seed{s}_per_class.csv' for s in SEEDS],
]


def load_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources(runs_dir, reports_dir):
    summary = load_json(reports_dir / 'cnn_summary.json')
    if not (summary.get('model') == 'cnn' and summary.get('experiment_id') == 'C01'
            and summary.get('official_tensorflow_run') is True
            and summary.get('backend') == 'tensorflow'
            and summary.get('seeds') == SEEDS and summary.get('seed_count') == 3
            and summary.get('test_evaluated') is False):
        raise ValueError('Not a formal TensorFlow C01 summary for three fixed seeds')
    records = {r['seed']: r for r in summary['runs']}
    if set(records) != set(SEEDS):
        raise ValueError('Summary runs do not match three fixed seeds')
    for name in REPORT_FILES:
        if not (reports_dir / name).is_file():
            raise FileNotFoundError(reports_dir / name)
    for seed in SEEDS:
        path = runs_dir / f'C01_seed{seed}'
        for name in RUN_FILES:
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
        config = load_json(path / 'config.json')
        meta = load_json(path / 'metadata.json')
        metrics = load_json(path / 'metrics.json')
        status = load_json(path / 'status.json')
        check = load_json(path / 'fresh_process_verification.json')
        reload_check = load_json(path / 'reload_check.json')
        runtime = meta.get('environment', {})
        verified_runs = check.get('runs', [])
        verified = (check.get('passed') is True and isinstance(verified_runs, list)
                    and len(verified_runs) == 1 and isinstance(verified_runs[0], dict)
                    and verified_runs[0].get('passed') is True
                    and verified_runs[0].get('count') == 1775
                    and verified_runs[0].get('test_evaluated') is False
                    and verified_runs[0].get('run_id') == path.name)
        if (status.get('status') != 'complete' or config.get('seed') != seed
                or config.get('model_key') != 'cnn'
                or config.get('experiment_id') != 'C01'
                or meta.get('experiment_id') != 'C01'
                or meta.get('model_input_shape') != [128, 9]
                or meta.get('output_shape') != [6]
                or not runtime.get('tensorflow')
                or not verified
                or not reload_check.get('passed')
                or metrics.get('smoke') is not False
                or metrics.get('test_evaluated') is not False
                or metrics.get('validation_count') != 1775
                or metrics.get('parameters') != 85574
                or records[seed].get('backend') != 'tensorflow'
                or records[seed].get('fresh_process_verified') is not True
                or records[seed].get('model_reload_verified') is not True):
            raise ValueError(f'Unverified / non-formal TensorFlow run: {path}')
        for name, key in [('model.keras', 'model_sha256'),
                          ('preprocess.npz', 'preprocess_sha256')]:
            if sha256(path / name) != meta.get(key):
                raise ValueError(f'Hash mismatch: {path / name}')
        for name in ('accuracy', 'macro_f1'):
            if abs(records[seed][name] - metrics[name]) > 1e-12:
                raise ValueError(f'Published summary differs from saved run: {seed}/{name}')
        with (path / 'validation_predictions.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1775:
            raise ValueError(f'Incorrect validation prediction count: {path}')
    return summary


def paper_text(summary):
    rows = []
    for r in summary['runs']:
        rows.append(f"| {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['epochs_ran']} | {r['best_epoch']} |")
    rows.append(f"| **평균 ± 표본 SD** | **{summary['accuracy_mean']:.4f} ± {summary['accuracy_sample_sd']:.4f}** | **{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_sample_sd']:.4f}** | — | — |")
    return ('# CNN 정식 TensorFlow 검증 결과 (자동 생성)\n\n'
            '> **TensorFlow 정식 실행 및 독립 새 프로세스 재검증 확인 후 생성되었습니다.** '
            '공식 test 결과가 아니라 고정 피험자 분할 S0의 validation 결과입니다.\n\n'
            '## 실험 조건\n\n'
            '- 데이터: UCI HAR, 128 시점 × 9 채널, 6개 행동.\n'
            '- 학습 5,577개, 검증 1,775개, 피험자 비중첩 분할.\n'
            '- 모델: 1D CNN, 85,574 학습 파라미터.\n'
            '- seed: 2026, 2027, 2028.\n'
            '- Adam(lr=0.001), batch=64, 최대 40 epoch, val_loss 조기 종료 patience=6.\n'
            '- 백엔드: TensorFlow (학습 폴더 metadata.json의 실제 버전 참조).\n'
            '- 제출한 평가 수치는 validation 기준이며 test 집합은 평가하지 않았습니다.\n\n'
            '## 검증 결과\n\n'
            '| Seed | Accuracy | Macro F1 | 종료 Epoch | 최저 검증 손실 Epoch |\n'
            '|---|---:|---:|---:|---:|\n'
            + '\n'.join(rows) + '\n\n'
            '세 seed의 표본 표준편차는 **고정된 분할에서 초기화 seed 변화**를 나타내며 '
            '새로운 피험자 집단에 대한 통계적 신뢰구간이 아닙니다.\n\n'
            '## 재현 및 증거\n\n'
            '- `reports/cnn/cnn_summary.json`: 세 seed 검증 통계\n'
            '- `reports/cnn/cnn_runs.csv`: 개별 실행 기록\n'
            '- `reports/cnn/cnn_confusion.png`: 검증 혼동행렬\n'
            '- `reports/cnn/cnn_learning_curves.png`: 학습·검증 손실\n'
            '- `artifacts/cnn_tensorflow/C01_seedXXXX/`: 실제 모델·예측·평가·재검증 기록\n'
            '- `artifacts/cnn_tensorflow/manifest.json`: 파일별 SHA-256\n\n'
            '작성자 확인 없이 예비 Keras/Torch 결과와 평균을 섞어 비교하지 마십시오.\n')


def publish(runs_dir, reports_dir, artifacts_dir, paper_path, replace=False):
    summary = verify_sources(runs_dir, reports_dir)
    if artifacts_dir.exists() and not replace:
        raise FileExistsError(f'{artifacts_dir} exists; use --replace to regenerate')
    artifacts_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cnn_publish_', dir=artifacts_dir.parent) as temp:
        stage = Path(temp) / artifacts_dir.name
        stage.mkdir()
        manifest = {'experiment_id': 'C01', 'backend': 'tensorflow',
                    'official_test_evaluated': False, 'seeds': SEEDS, 'files_sha256': {}}
        for seed in SEEDS:
            name = f'C01_seed{seed}'
            (stage / name).mkdir()
            for filename in RUN_FILES:
                target = stage / name / filename
                shutil.copy2(runs_dir / name / filename, target)
                manifest['files_sha256'][f'{name}/{filename}'] = sha256(target)
        (stage / 'README.md').write_text(
            '# CNN C01 TensorFlow 검증 완료 모델\n\n'
            '세 seed의 `.keras` 파일과 각각의 동일 검증 예측·전처리·metadata를 보관합니다. '
            '`manifest.json`의 SHA-256을 확인하세요. 공식 test는 사용하지 않았습니다.\n'
            '동일 입력이 필요하면 프로젝트 루트에서 다음 명령으로 재검증할 수 있습니다.\n\n'
            '```bash\npython -m src.verify_bundle --processed-data data/processed_data.npz '
            '--preprocess-stats data/preprocess.npz '
            '--run-directory artifacts/cnn_tensorflow/C01_seed2026 '
            '--report reports/cnn/reverify_seed2026.json\n```\n', encoding='utf-8')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if artifacts_dir.exists():
            backup = artifacts_dir.with_name(artifacts_dir.name + '.previous')
            if backup.exists():
                raise FileExistsError(f'Backup path exists: {backup}')
            artifacts_dir.rename(backup)
            try:
                shutil.copytree(stage, artifacts_dir)
            except Exception:
                artifacts_dir.rename(artifacts_dir.with_name(artifacts_dir.name + '.failed'))
                backup.rename(artifacts_dir)
                raise
            shutil.rmtree(backup)
        else:
            shutil.copytree(stage, artifacts_dir)
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(paper_text(summary), encoding='utf-8')
    print('TensorFlow final deliverables published:', artifacts_dir, reports_dir, paper_path)
    print('Commit with: git add artifacts/cnn_tensorflow reports/cnn docs/team-paper/contributions/03_cnn/')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', type=Path, default=ROOT / 'runs/cnn')
    p.add_argument('--reports-dir', type=Path, default=ROOT / 'reports/cnn')
    p.add_argument('--artifacts-dir', type=Path, default=ROOT / 'artifacts/cnn_tensorflow')
    p.add_argument('--paper-path', type=Path, default=ROOT / 'docs/team-paper/contributions/03_cnn/final_tensorflow_results.md')
    p.add_argument('--replace', action='store_true')
    args = p.parse_args()
    publish(args.runs_dir, args.reports_dir, args.artifacts_dir, args.paper_path, args.replace)


if __name__ == '__main__':
    main()
