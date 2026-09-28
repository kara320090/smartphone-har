"""Demonstrate a lossless round trip of 200 fit rows through TFRecord."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from .adapters.uci_reference import load_dataset
from .runtime import file_sha256, write_json


def roundtrip(samples, indices, record_path):
    ids = np.asarray(indices["fit"][:200])
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("X_seq", "X_features", "y", "subject", "sample_id")
    with tf.io.TFRecordWriter(str(record_path)) as writer:
        for index in ids:
            features = {}
            for name in fields:
                value = tf.convert_to_tensor(samples[name][index])
                encoded = tf.io.serialize_tensor(value).numpy()
                features[name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded]))
            writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
    dtypes = dict(X_seq=tf.float32, X_features=tf.float32, y=tf.int64,
                  subject=tf.int64, sample_id=tf.string)
    seen = 0
    for row, record in enumerate(tf.data.TFRecordDataset(str(record_path))):
        decoded = tf.io.parse_single_example(record, {k: tf.io.FixedLenFeature([], tf.string) for k in fields})
        for name in fields:
            actual = tf.io.parse_tensor(decoded[name], out_type=dtypes[name]).numpy()
            expected = samples[name][ids[row]]
            if name == "sample_id":
                actual = actual.decode("utf-8")
            np.testing.assert_array_equal(actual, expected)
        seen += 1
    if seen != len(ids):
        raise AssertionError("TFRecord count mismatch")
    return {"passed": True, "count": seen, "fields": list(fields),
            "comparison": "exact equality of values, shapes, labels, IDs and subjects",
            "source_split": "fit", "file_sha256": file_sha256(record_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--record-path", type=Path, default=Path("work/fit_200.tfrecord"))
    parser.add_argument("--report", type=Path, default=Path("reports/tfrecord_check.json"))
    args = parser.parse_args()
    samples, indices = load_dataset(args.data_root)
    result = roundtrip(samples, indices, args.record_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(result)
