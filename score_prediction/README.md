# Historical Score Prediction Experiment

Recovered data and source code for the score prediction task in Section 5.2 of
[KokoroChat (ACL 2025)](https://aclanthology.org/2025.acl-long.608/).
These files were recovered from the original `llama-3-P` experiment directory
on November 17, 2026. Historical JSON and Python files are preserved byte for
byte; JSON files are distributed with lossless gzip compression.

## Data and fixed test split

| File | Records | Contents |
| --- | ---: | --- |
| `datasets/test_data_P.json.gz` | 658 | Fixed test inputs and all 20 human feedback scores |
| `datasets/finetune_data_P.json.gz` | 5,931 | Training/validation pool, with all 20 target scores |
| `datasets/finetune_data_P_noneed.json.gz` | 6,589 | Historical complete pool before test exclusion |
| `human_scores.csv` | 6,589 | Human scores and split membership in a readable table |
| `split_indices.json` | 6,589 indices | Exact ordered membership of the test set and training/validation pool |

The test set contains 658 unique internal indices. It has zero overlap with
the 5,931-record pool; their union covers all 6,589 records. Do not randomly
resplit the public dataset when reproducing this test set.

`index` is the historical internal index. It is **not** a public filename ID in
`../kokorochat_dialogues/`; do not use it as one. A complete verified crosswalk
has not been recovered. The released inputs themselves identify the exact
evaluation examples without needing a crosswalk.

Each test/pool record has `instruction`, `input`, `output_all`, and
`output_1` through `output_20`. Scores range from 0 to 5. `output_all` contains
20 labeled item scores. The complete-pool archive instead uses `output`, which
also includes the aggregate `点数` field. CSV headers preserve the Japanese
dimension names. All 20 scores were checked against the complete pool.

The historical `input` is preprocessed experiment text. Some inputs contain
only a prefix of the corresponding public dialogue, so they must not be
described as verbatim, complete public transcripts. Use these preserved inputs
to match the historical evaluation conditions.

## Verify and extract

From this directory, using Python 3 and its standard library:

```bash
python3 verify_release.py
python3 verify_release.py --extract
```

The second command also decompresses archives next to their `.gz` files.
Existing decompressed files are accepted only if their bytes match the archive.
`source_manifest.json` records SHA-256 checksums for the archives, their
original uncompressed bytes, original scripts, and checkpoint metadata.

## Recovered original code

All files in `scripts/` are unchanged historical snapshots, not rewritten
training implementations:

| Script | Historical purpose |
| --- | --- |
| `finetune_AP_4seed.py` | LoRA training with train/validation seeds `[2, 12, 22, 32]` and `output_all` targets |
| `finetune_AP_.py` | LoRA configuration sweep; current snapshot selects `adamw_8bit`, learning rate `1e-4`, warmup 100 |
| `train_para_tune.py` | Additional LoRA hyperparameter sweep |
| `finetune_AP.py` | Earlier multi-configuration training variant without an attached PEFT configuration |
| `finetune_P.py` | Earlier single-dimension (`output_1`) LoRA training variant |
| `generate_lora.py` | Evaluate multiple saved adapters/checkpoints on the fixed test set |
| `generate_llama.py` | Base-model score prediction on the fixed test set |
| `generate_with_ft_P.py` | Adapter-based score prediction |
| `merge_model.py` | Merge a saved adapter into the base model |

Paths such as `datasets/finetune_data_P.json` are relative to this directory.
Extract the archives before using the scripts. Training requires a CUDA setup,
access to `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3`, and compatible
versions of PyTorch, Transformers, Datasets, PEFT, TRL, scikit-learn, NumPy,
bitsandbytes, and Weights & Biases. A verified historical dependency lockfile
was not recovered. Training scripts can launch W&B sweeps and costly training
at import time; inspect them and configure your own accounts before execution.

### Historical limitations

- The initial 658-record test-selection seed/code has not been recovered.
  The exact fixed test set is supplied instead.
- The scripts split the 5,931-record pool with `test_size=0.11`; the fixed-seed
  variants use `random_state=42`. This produces 5,278 training and 653
  validation records, approximately the paper's 8:1:1 description. Comments
  saying "80%/20%" do not describe the actual code.
- Several snapshots construct `bnb_config` but do not pass
  `quantization_config` to model loading. Do not assume these snapshots activate
  4-bit training solely because the paper describes QLoRA.
- The multi-seed snapshot loads the base model outside its seed loop. It is
  preserved as recovered, not certified as independent fresh initialization
  for every seed.
- `generate_lora.py` currently selects directory names starting with
  `-checkpoint-`. Standard `checkpoint-249` directories will not match that
  filter unless the script is adapted.
- The exact source revision and five-seed-to-checkpoint mapping for every
  reported run have not been established. No training or GPU inference was
  rerun for this release.

## Result provenance

`reference_result/` contains the original prediction JSON for
`results_optim_adamw_torch_fused_lr_0.0002_ws_100/checkpoint-249`.
It covers the same 658 test indices and all 13,160 item predictions. Recomputing
from that file gives:

| ACC | Soft ACC (absolute error <= 1) | MAE |
| ---: | ---: | ---: |
| 36.41% | 82.58% | 0.8413 |

These rounded values match **Seed 1 in Table 6** of the paper. This establishes
a direct link between the preserved test inputs/results and the publication;
it is not a claim that all paper runs have been fully reproduced.

`checkpoint_metadata/` contains the original `adapter_config.json` and
`trainer_state.json` for this checkpoint. The latter records epoch 3.0 and
global step 249. Model weights are not included in Git. The published scoring
model is available at
[UEC-InabaLab/Llama-3.1-KokoroChat-ScorePrediction](https://huggingface.co/UEC-InabaLab/Llama-3.1-KokoroChat-ScorePrediction).
Identity between that published model and this particular checkpoint has not
been verified in this release.

The sibling historical `test_results_P.json` and `test_results_P_all.json`
use a different 658-record split with only 73 indices in common with this
test set. They are intentionally not part of this release.

## License

See the repository's [license](../LICENSE) and [dataset documentation](../README.md).
This directory does not introduce a new license or alter the existing terms.
