# Score Prediction

Data, training code, and prediction results for the score prediction task in
Section 5.2 of [KokoroChat (ACL 2025)](https://aclanthology.org/2025.acl-long.608/).
The task predicts 20 human feedback scores, each ranging from 0 to 5, from a
counseling dialogue input.

## Data and fixed test split

| File | Records | Contents |
| --- | ---: | --- |
| `datasets/test_data_P.json.gz` | 658 | Fixed test inputs and all 20 human feedback scores |
| `datasets/finetune_data_P.json.gz` | 5,931 | Training/validation pool, with all 20 target scores |
| `datasets/finetune_data_P_noneed.json.gz` | 6,589 | Complete scoring dataset before test exclusion |
| `human_scores.csv` | 6,589 | Human scores and split membership in a readable table |
| `split_indices.json` | 6,589 indices | Exact ordered membership of the test set and training/validation pool |

The fixed test set and training/validation pool are disjoint and together
cover all 6,589 records. `split_indices.json` provides their ordered indices.
`index` identifies records within these experiment files; the public dialogue
files in `../kokorochat_dialogues/` use a separate ID namespace.

Each test/pool record has `instruction`, `input`, `output_all`, and
`output_1` through `output_20`. Scores range from 0 to 5. `output_all` contains
20 labeled item scores. The complete-pool archive instead uses `output`, which
also includes the aggregate `点数` field. CSV headers preserve the Japanese
dimension names.

`input` contains the preprocessed text used in the experiment, including
dialogue prefixes where applicable. The JSON archives preserve the original
experiment files with lossless gzip compression.

## Verify and extract

From this directory, using Python 3 and its standard library:

```bash
python3 verify_release.py
python3 verify_release.py --extract
```

The first command checks file hashes, split membership, all 20 human scores,
and the reference prediction metrics. The second also decompresses the JSON
archives next to their `.gz` files. `source_manifest.json` provides SHA-256
checksums for the archives, uncompressed JSON files, scripts, and metadata.

## Training and inference code

`scripts/` contains the original experiment code:

| Script | Purpose |
| --- | --- |
| `finetune_AP_4seed.py` | LoRA training with train/validation seeds `[2, 12, 22, 32]` and `output_all` targets |
| `finetune_AP_.py` | LoRA configuration sweep with `adamw_8bit`, learning rate `1e-4`, and warmup 100 |
| `train_para_tune.py` | Additional LoRA hyperparameter sweep |
| `finetune_AP.py` | Full-model training with multiple configurations |
| `finetune_P.py` | Single-dimension (`output_1`) LoRA training |
| `generate_lora.py` | Evaluate multiple saved adapters/checkpoints on the fixed test set |
| `generate_llama.py` | Base-model score prediction on the fixed test set |
| `generate_with_ft_P.py` | Adapter-based score prediction |
| `merge_model.py` | Merge a saved adapter into the base model |

The base model is `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3`.
The train/validation scripts use `test_size=0.11` on the 5,931-record pool,
giving 5,278 training and 653 validation records. The single-split scripts use
`random_state=42`; `finetune_AP_4seed.py` supplies the multi-seed settings.

Paths such as `datasets/finetune_data_P.json` are relative to this directory.
Extract the archives before running the scripts. Dependencies include PyTorch,
Transformers, Datasets, PEFT, TRL, scikit-learn, NumPy, bitsandbytes, and
Weights & Biases. Model paths, checkpoint directory names, CUDA settings, and
W&B configuration are specified in the scripts.

## Prediction results

`reference_result/` contains the predictions from
`results_optim_adamw_torch_fused_lr_0.0002_ws_100/checkpoint-249`.
It covers the same 658 test indices and all 13,160 item predictions. Recomputing
from that file gives:

| ACC | Soft ACC (absolute error <= 1) | MAE |
| ---: | ---: | ---: |
| 36.41% | 82.58% | 0.8413 |

These values match **Seed 1 in Table 6** of the paper.

`checkpoint_metadata/` provides `adapter_config.json` and
`trainer_state.json` for this checkpoint. The latter records epoch 3.0 and
global step 249. The published scoring model is available at
[UEC-InabaLab/Llama-3.1-KokoroChat-ScorePrediction](https://huggingface.co/UEC-InabaLab/Llama-3.1-KokoroChat-ScorePrediction).

## License

See the repository's [license](../LICENSE) and [dataset documentation](../README.md).
