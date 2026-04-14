# FoodMatchMTL

This directory contains the implementation of the proposed FoodMatchMTL method together with the experiment configurations and result files used in `4.experiment`.

## Contents

- `train.py`
  - Training and evaluation CLI.
- `build_taxonomy_graph.py`
  - Preprocessing utilities for taxonomy-related assets.
- `clean_ingredient_alias_dict.py`
  - Cleaning script for the ingredient alias dictionary.
- `expand_experiments.py`
  - Utility for expanding experiment settings.
- `run_experiments_from_yaml.sh`
  - Batch execution script from YAML configurations.
- `run_proj_head_ablation.sh`
  - Script for the task-combination sweep used in `4.experiment`.
- `experiments/`
  - Representative experiment configurations.
- `train_data/`
  - Lightweight auxiliary data files required by the copied scripts.
- `results/`
  - Executed results required for `4.experiment`.
  - Each run includes `config.json` and `eval_result.json`.
  - Task-sweep summary files for each encoder are also included.

## Notes

- The selected runs used in the paper and the runs referenced by the ablation analysis are included.
- Large taxonomy assets and full experiment logs are not included.
- A `src/ours_v2` symlink is kept in the repository for compatibility with existing notebooks and summary JSON files.
