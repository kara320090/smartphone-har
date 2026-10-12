import json
import pytest
from src.rnn_analysis import analyze


def test_non_rnn_evidence_cannot_enter_rnn_report(tmp_path):
    run = tmp_path / 'runs' / 'C01_seed2026'
    run.mkdir(parents=True)
    (run / 'metadata.json').write_text('{}')
    (run / 'config.json').write_text(json.dumps({'model_key': 'cnn', 'experiment_id': 'C01'}))
    with pytest.raises(ValueError, match='RNN 전용'):
        analyze(run.parent, tmp_path / 'report')
