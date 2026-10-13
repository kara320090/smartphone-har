"""Recompute the joint development table from each contributor's saved predictions.

No training or official test access. LSTM evidence is supplied by its existing
Release (runs/lstm by default); the other contributors' predictions are in Git.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def scores(rows):
    matrix = [[0] * 6 for _ in range(6)]
    for row in rows:
        matrix[int(row['true_label'])][int(row['predicted_label'])] += 1
    f1 = []
    for i in range(6):
        denominator = sum(matrix[i]) + sum(r[i] for r in matrix)
        f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    return sum(matrix[i][i] for i in range(6)) / len(rows), statistics.mean(f1), matrix


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lstm-runs', type=Path, default=Path('runs/lstm'))
    parser.add_argument('--output', type=Path, default=Path('reports/team_comparison'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    lstm = args.lstm_runs.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ('MLP', 'M01', '한윤섭', repo/'reports/mlp_yoonseop/run_evidence', 'baseline', 156230),
        ('CNN', 'C01', '최승빈', repo/'artifacts/cnn_tensorflow', 'baseline', 85574),
        ('RNN', 'R01', '유재윤', repo/'reports/rnn/run_evidence', 'baseline', 1542),
        ('LSTM', 'L01', '이봉헌', lstm, 'baseline', 21222),
        ('RNN', 'R02', '유재윤', repo/'reports/rnn/run_evidence', 'individual_tuning', 5126),
    ]
    # Independent counts from the checked builders, including layer biases.
    parameter_counts = {
        'M01': (1152+1)*128 + (128+1)*64 + (64+1)*6,
        'C01': (5*9+1)*64 + (3*64+1)*128 + (3*128+1)*128 + (128+1)*64 + (64+1)*6,
        'R01': 32*(9+32+1) + (32+1)*6,
        'L01': 4*64*(9+64+1) + (64+1)*32 + (32+1)*6,
        'R02': 64*(9+64+1) + (64+1)*6,
    }
    assert all(parameter_counts[config_id] == count for _, config_id, _, _, _, count in specs)
    expected_channels = ['body_acc_' + x for x in 'xyz'] + ['body_gyro_' + x for x in 'xyz'] + ['total_acc_' + x for x in 'xyz']
    expected_classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    runs, subjects, provenance, evidence = [], [], [], []
    gold, first_split = None, None
    for model, config_id, owner, directory, role, expected_parameters in specs:
        for seed in (2026, 2027, 2028):
            run_id = f'{config_id}_seed{seed}'
            base = directory / run_id
            metadata = load_json(base/'metadata.json')
            config = load_json(base/'config.json')
            split = load_json(base/'split.json')
            metrics = load_json(base/'metrics.json')
            reload_check = load_json(base/'reload_check.json')
            fresh_check = load_json(base/'fresh_process_verification.json')
            assert load_json(base/'status.json')['status'] == 'complete', run_id
            assert reload_check['passed'] and fresh_check['passed'], run_id
            assert not metrics['test_evaluated'] and not split['test_evaluated']
            assert metadata['model_input_shape'] == [128, 9]
            assert metadata['channels'] == expected_channels and metadata['classes'] == expected_classes
            assert metadata['sample_row_number_base'] == 0 and metadata['label_offset_from_original'] == -1
            prov = metadata['data_provenance']
            assert prov['standardization_applications'] == 1 and not prov['statistics_refitted']
            assert not prov['test_arrays_loaded'] and prov['statistics_match_full_fit']
            shared_provenance = {k: prov[k] for k in ('data_sha256', 'source_stats_sha256', 'fit_row_binding_sha256', 'validation_row_binding_sha256')}
            if provenance:
                assert shared_provenance == provenance[0], run_id
            provenance.append(shared_provenance)
            assert len(split['fit_ids']) == 5577 and len(split['validation_ids']) == 1775
            assert set(split['fit_ids']).isdisjoint(split['validation_ids'])
            assert set(split['fit_subjects']).isdisjoint(split['validation_subjects'])
            assert sorted(split['validation_subjects']) == [1, 6, 14, 21, 23]
            if first_split is None:
                first_split = split
            else:
                assert split == first_split, run_id
            common = {'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 64, 'epochs': 40, 'patience': 6, 'monitor': 'val_loss', 'restore_best_weights': True, 'threads': 4}
            assert config['seed'] == seed
            for key, value in common.items():
                assert config[key] == value, (run_id, key, config.get(key))
            assert not config['smoke'] and metrics['parameters'] == expected_parameters
            predictions_path = base/'validation_predictions.csv'
            rows = read_csv(predictions_path)
            assert len(rows) == 1775
            binding = [(r['sample_id'], int(r['subject']), int(r['true_label'])) for r in rows]
            assert len({r[0] for r in binding}) == 1775
            assert [r[0] for r in binding] == split['validation_ids']
            if gold is None:
                gold = binding
            else:
                assert binding == gold, run_id
            for row in rows:
                true, pred = int(row['true_label']), int(row['predicted_label'])
                assert 0 <= true < 6 and 0 <= pred < 6
                probabilities = [float(row[f'probability_{i}']) for i in range(6)]
                logits = [float(row[f'logit_{i}']) for i in range(6)]
                assert all(math.isfinite(x) for x in probabilities + logits)
                assert all(0 <= p <= 1 for p in probabilities)
                assert abs(sum(probabilities) - 1) <= 1e-6
                assert probabilities.index(max(probabilities)) == pred == logits.index(max(logits))
                exp = [math.exp(x - max(logits)) for x in logits]
                assert all(abs(p - e/sum(exp)) <= 1e-5 for p, e in zip(probabilities, exp))
            accuracy, macro_f1, confusion = scores(rows)
            assert abs(accuracy - metrics['accuracy']) < 1e-12
            assert abs(macro_f1 - metrics['macro_f1']) < 1e-12
            assert confusion == metrics['confusion_matrix']
            runs.append({'model': model, 'config_id': config_id, 'role': role, 'owner': owner, 'seed': seed, 'validation_count': len(rows), 'parameters': expected_parameters, 'dropout': config['dropout'], 'accuracy': accuracy, 'macro_f1': macro_f1, 'best_epoch': metrics['best_epoch'], 'epochs_ran': metrics['epochs_ran'], 'python': metadata['environment']['python'], 'tensorflow': metadata['environment']['tensorflow'], 'keras': metadata['environment']['keras'], 'processor': metadata['environment']['processor']})
            for subject in [1, 6, 14, 21, 23]:
                subset = [r for r in rows if int(r['subject']) == subject]
                acc, f1, _ = scores(subset)
                subjects.append({'model': model, 'config_id': config_id, 'role': role, 'seed': seed, 'subject': subject, 'count': len(subset), 'accuracy': acc, 'macro_f1_six_classes': f1})
            for filename in ('config.json', 'metadata.json', 'split.json', 'metrics.json', 'reload_check.json', 'fresh_process_verification.json', 'validation_predictions.csv'):
                path = base / filename
                evidence.append({'run_id': run_id, 'file': path.relative_to(repo).as_posix() if path.is_relative_to(repo) else f'external_lstm/{run_id}/{filename}', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = []
    for model, config_id, owner, _, role, parameters in specs:
        group = [r for r in runs if r['config_id'] == config_id]
        summary.append({'model': model, 'config_id': config_id, 'role': role, 'owner': owner, 'repetitions': 3, 'parameters': parameters, 'dropout': group[0]['dropout'], 'accuracy_mean': statistics.mean(r['accuracy'] for r in group), 'accuracy_sample_sd': statistics.stdev(r['accuracy'] for r in group), 'macro_f1_mean': statistics.mean(r['macro_f1'] for r in group), 'macro_f1_sample_sd': statistics.stdev(r['macro_f1'] for r in group), 'evaluation_split': 'development validation', 'test_evaluated': False})
    # Check the four original published summaries as well as the selected RNN row.
    published = {
        'M01': read_csv(repo/'reports/mlp_yoonseop/summary.csv')[0],
        'C01': load_json(repo/'reports/cnn/cnn_summary.json'),
        **{r['experiment']: {**r, 'accuracy_sample_sd': r['accuracy_sd'], 'macro_f1_sample_sd': r['macro_f1_sd']} for r in read_csv(repo/'reports/rnn/summary.csv')},
        **{r['experiment']: r for r in read_csv(repo/'reports/lstm/summary.csv')},
    }
    for row in summary:
        for key in ('accuracy_mean', 'accuracy_sample_sd', 'macro_f1_mean', 'macro_f1_sample_sd'):
            assert abs(row[key] - float(published[row['config_id']][key])) < 1e-12
    paired = []
    for a, b in itertools.combinations(('M01', 'C01', 'R01', 'L01'), 2):
        for seed in (2026, 2027, 2028):
            x = next(r for r in runs if r['config_id'] == a and r['seed'] == seed)
            y = next(r for r in runs if r['config_id'] == b and r['seed'] == seed)
            paired.append({'config_a': a, 'config_b': b, 'seed': seed, 'macro_f1_a_minus_b': x['macro_f1'] - y['macro_f1'], 'accuracy_a_minus_b': x['accuracy'] - y['accuracy']})
    write_csv(output/'runs_recomputed.csv', runs)
    write_csv(output/'baseline_summary.csv', [r for r in summary if r['role'] == 'baseline'])
    write_csv(output/'supplementary_summary.csv', [r for r in summary if r['role'] != 'baseline'])
    write_csv(output/'paired_seed_differences.csv', paired)
    write_csv(output/'subject_metrics.csv', subjects)
    audit = {'date_korea': '2026-10-13', 'source_commit': source_commit, 'passed': True, 'baseline_runs_checked': 12, 'supplementary_runs_checked': 3, 'prediction_rows_checked': len(runs) * 1775, 'split_identical': True, 'ids_subjects_labels_order_identical': True, 'data_provenance': provenance[0], 'metrics_recomputed_and_published_summaries_matched': True, 'saved_reload_records_passed': True, 'fresh_model_reload_performed_in_this_audit': False, 'test_evaluated': False, 'model_selection_scope': 'R01 retained for the baseline draft; R02 reported separately as the contributor-selected tuning result. Team final comparison protocol still requires agreement.', 'limitations': ['validation used for checkpoint selection', 'one fixed subject split and three seeds', 'model capacity and Dropout differ', 'different PCs and Python/NumPy versions; training-time ranking omitted', 'RNN tuning budget differs from MLP/CNN', 'official test not evaluated'], 'evidence': evidence}
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'passed': True, 'baseline_runs': 12, 'supplementary_runs': 3, 'prediction_rows': len(runs) * 1775, 'source_commit': source_commit, 'output': str(output)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
