# FoodMatchMTL: Ingredient-to-Food Composition Table Matching via Taxonomy-Aware Multi-Task Learning with Structured Decomposition

This repository contains the code, selected data files, experiment outputs, and notebook used for the experiments described in `4.experiment`.

## Overview

Ingredient matching links recipe ingredient names to entries in a food composition table for nutrient estimation. This task is challenging because recipe texts show large descriptive variation, biologically related ingredients may have very different surface forms, and text similarity alone does not reliably capture biological proximity.

FoodMatchMTL addresses this problem as a structured retrieval task with taxonomy-aware multi-task learning. The framework combines three complementary objectives: ID retrieval as the main task, ingredient category classification to model coarse semantic structure, and species-aware contrastive learning to incorporate biological relationships. Across multiple encoder language models, this formulation improves retrieval accuracy and consistently outperforms single-task fine-tuning.

## Repository Structure

- `data/`: core data files used by the copied experiment scripts and reports.
- `src/baseline/`: baseline implementations and executed result files used in `4.experiment`.
- `src/FoodMatchMTL/`: training code, experiment configurations, and executed FoodMatchMTL results used in `4.experiment`.
- `notebook/4_experiment_report.ipynb`: notebook for inspecting the main comparison results and ablation results from `4.experiment`.
- `notebook/artifacts/`: exported artifacts such as `table1_comparison.csv` and `figure2_ablation.png`.

## Where To Start

- Main comparison CSV: [notebook/artifacts/table1_comparison.csv](./notebook/artifacts/table1_comparison.csv)
- Experiment notebook: [notebook/4_experiment_report.ipynb](./notebook/4_experiment_report.ipynb)
- Ablation figure: [notebook/artifacts/figure2_ablation.png](./notebook/artifacts/figure2_ablation.png)

## Notebook Notes

- The notebook is adapted from `KES2026_report.ipynb` to match this repository layout.
- The sections corresponding to `4.experiment` can be inspected in this layout.
- The evaluation results are displayed in the notebook as pandas DataFrames.
- Some later notebook cells may still assume additional artifacts that are not included here.
