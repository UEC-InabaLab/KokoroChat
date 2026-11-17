"""Verify the recovered release without importing or executing training code."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def numbers(text):
    return [int(value) for value in re.findall(r'= *([0-9]+)', text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    data = {}
    for name, entry in manifest['files'].items():
        path = ROOT / name
        raw = path.read_bytes()
        check(hashlib.sha256(raw).hexdigest() == entry['sha256'], f'Checksum mismatch: {name}')
        if path.suffix == '.gz':
            raw = gzip.decompress(raw)
            check(hashlib.sha256(raw).hexdigest() == entry['sha256_uncompressed'],
                  f'Uncompressed checksum mismatch: {name}')
            data[path.stem] = json.loads(raw)
            if args.extract:
                target = path.with_suffix('')
                if target.exists():
                    check(target.read_bytes() == raw, f'Refusing to replace differing file: {target}')
                else:
                    with target.open('xb') as handle:
                        handle.write(raw)
        elif path.suffix == '.py':
            ast.parse(raw, filename=str(path))
    full = {row['index']: row for row in data['finetune_data_P_noneed.json']}
    pool = data['finetune_data_P.json']
    test = data['test_data_P.json']
    splits = json.loads((ROOT / 'split_indices.json').read_text())
    check(len(full) == len(data['finetune_data_P_noneed.json']) == 6589, 'Invalid complete pool')
    for rows, name, count in [(pool, 'train_validation_pool', 5931), (test, 'test', 658)]:
        ids = [row['index'] for row in rows]
        check(len(ids) == len(set(ids)) == count, f'Invalid IDs: {name}')
        check(ids == splits[name], f'Incorrect ordered split manifest: {name}')
        for row in rows:
            check(row['input'] == full[row['index']]['input'], 'Input mismatch with full pool')
            scores = [int(row[f'output_{i}'].rsplit('=', 1)[1]) for i in range(1, 21)]
            check(all(0 <= value <= 5 for value in scores), 'Score outside 0..5')
            check(scores == numbers(row['output_all']), 'Inconsistent target fields')
            check(scores == numbers(full[row['index']]['output'])[1:], 'Mismatch with full human scores')
    pool_ids, test_ids = set(splits['train_validation_pool']), set(splits['test'])
    check(not pool_ids & test_ids and pool_ids | test_ids == set(full), 'Invalid split partition')
    expected = {row['index']: row for row in pool + test}
    with (ROOT / 'human_scores.csv').open(newline='') as handle:
        table = list(csv.reader(handle))
    check(len(table) == 6590, 'Incorrect score CSV length')
    check(len({row[0] for row in table[1:]}) == 6589, 'Duplicate score CSV ID')
    for row in table[1:]:
        check(row[1] == ('test' if row[0] in test_ids else 'train_validation_pool'), 'Wrong CSV split')
        check([int(value) for value in row[2:]] == numbers(expected[row[0]]['output_all']), 'CSV score mismatch')
    result_name = next(name for name in data if name.startswith('test_results_'))
    results = data[result_name]
    check(len(results) == len({row['index'] for row in results}) == 658, 'Invalid reference result IDs')
    check({row['index'] for row in results} == test_ids, 'Reference result uses a different test set')
    errors = []
    for row in results:
        target = expected[row['index']]
        check(row['input'] == target['input'], 'Reference input mismatch')
        original, prediction = numbers(row['output_original']), numbers(row['output_model'])
        check(original == numbers(target['output_all']), 'Reference human score mismatch')
        check(len(original) == len(prediction) == 20, 'Incomplete reference scores')
        check(all(0 <= value <= 5 for value in prediction), 'Reference prediction outside 0..5')
        errors.extend(abs(left - right) for left, right in zip(original, prediction))
    acc = 100 * sum(error == 0 for error in errors) / len(errors)
    soft = 100 * sum(error <= 1 for error in errors) / len(errors)
    mae = sum(errors) / len(errors)
    check((round(acc, 2), round(soft, 2), round(mae, 4)) == (36.41, 82.58, 0.8413),
          'Reference metrics do not match paper Seed 1')
    print(f'PASS: 6589 unique records; 5931 pool + 658 test; overlap 0; all 20 human scores verified.')
    print(f'PASS: {len(errors)} reference predictions; ACC {acc:.2f}%; Soft ACC {soft:.2f}%; MAE {mae:.4f}.')
    print('PASS: original-file checksums and archived Python syntax verified; no training executed.')


if __name__ == '__main__':
    main()
