# RCT Rejection Sampling

This repository hosts the code and data to support the paper ["RCT Rejection Sampling for Causal Estimation Evaluation"](https://arxiv.org/abs/2307.15176) by Keith, Feldman, Jurgens, Bragg and Bhattacharya. TMLR, 2023. 

If you use this data or code, please cite our paper:

```
@article{keith2023rct,
  title={RCT Rejection Sampling for Causal Estimation Evaluation},
  author={Keith, Katherine A and Feldman, Sergey and Jurgens, David and Bragg, Jonathan and Bhattacharya, Rohit},
  journal={Transactions on Machine Learning Research},
  year={2023}
}

```

**Corresponding author**: Email Katie Keith, kak5@williams.edu

## Change Log 
`2025-05-23`
- Updated `requirements.in` with correct `scipy` version 
- Updated `scripts/subpopA_physics_medicine.ipynb` and `scripts/subpopB_engineering_business.ipynb` with new cells 13-16 which save to disk the confounded datasets used in our proof of concept pipeline. 
  - These saved datasets can now be found in `data/confounded/subpopA.pkl` and `data/confounded/subpopB.pkl`

## Repository outline

- `data/`
  - `s2_rcts_full.csv`
  - `subpopA_physics_medicine.csv`
  - `subpopB_engineering_business.csv`
- `causal_eval/`
  - `sampling.py` - hosts our RCT rejection sampling algorithm
  - `causal_inference_algorithms.py` - causal estimation algorithms (including cross fitting with cross validation)
  - `ml_utils.py` - machine learning utility functions used by the causal estimation algorithms
  - `utils.py` - other general utility functions for the proof of concept pipeline
- `scripts/` - scripts to reproduce the results in our paper
  - `synthetic_experiments.ipynb` - creates plots
  - `synthetic_with_ci_coverage.py` - Creates results for Table 2
  - `subpopA_physics_medicine.ipynb` - Results for real-world dataset, subpopulation A 
  - `subpopB_engineering_business.ipynb`- results for real-world dataset, subpopulation B 
  - `synthetic_confounding_plot.ipynb` - script for plot in Appendix I 
- `README.md`
- `requirements.in`
- `setup.py`

## Installation and Set-up

```
git clone git@github.com:kakeith/rct_rejection_sampling.git
cd rct_rejection_sampling/
conda create -y --name evalEnv python==3.9 -c conda-forge
conda activate evalEnv
pip install -r requirements.in
pip install -e .
```

## Data

#### `s2_rct_full.csv`

The full dataset from the RCT run on the Semantic Scholar platform. See paper for more details.

Columns:

- `t`: The treatment (binary). Takes value 0 if the user did not see treatment and 1 else.
- `y`: The outcome (binary). Takes value 0 if the user clicked on the link and 1 else.
- `x`: The text (string). The concatenated title and abstract separated by a new line.
- `corpus_paper_id`: The paper id corresponding to Semantic Scholar's publicly facing paper ids.
- `arxiv_id`: The arxiv id of the paper. Has value None if there is no arxiv_id
- `s2fieldsofstudy`: Assigned field of study from the Semantic Scholar field of study classifier (S2FOS).

#### `subpopA_physics_medicine.csv`

The subset of `s2_rct_full.csv` for which the `s2fieldsofstudy` column contains either Physics or Medicine. This is used as Subpopulation A in the paper's proof of concept pipeline.

Columns:

- `X`: The text (string). The concatenated title and abstract separated by a new line.
- `Y`: The outcome (binary). Takes value 0 if the user clicked on the link and 1 else.
- `T`: The treatment (binary). Takes value 0 if the user did not see treatment and 1 else.
- `C`: The structured confounder (binary). This corresponds to the `s2fieldsofstudy` column in the full dataset and is 0 if "Medicine" and 1 if "Physics"

#### `subpopB_engineering_business.csv`

The subset of `s2_rct_full.csv` for which the `s2fieldsofstudy` column contains either Engineering or Business. This is used as Subpopulation B in the paper's proof of concept pipeline.

Columns:

- `X`, `Y`, `T`: are the same variables defined above for `subpopA_physics_medicine.csv`
- `C`: 0 if "Business" and 1 if "Engineering"

## Reproducing results in the paper

### Synthetic experiments

For the synthetic experiments (Table 2), run

```
scripts/synthetic_with_ci_coverage.py
```

### Proof of concept (real-world RCT)

The following scripts are the same except for the input data `.csv` files. We include both jupyter notebooks in order to display the results. However, if you would like to replicate this pipeline on an additional subpopulation, you can copy and modify either script.

For the proof of concept pipeline results for Subpopulation A, Physics and Medicine as the structured confounders, in Figure 3 and Table 3 run

```
scripts/subpopA_physics_medicine.ipynb
```

For the proof of concept pipeline results for Subpopulation B, Engineering and Business as the structured confounders, in Figure 4 and Table 6 run

```
scripts/subpopB_engineering_business.ipynb
```

## Code reuse

### RCT rejection sampling

The core contribution of our paper is the RCT Rejection Sampling algorithm (Algorithm 1). Here's a sample use of that algorithm

```
python causal_eval/sampling.py
```

### Estimation pipeline

Here's an example of using our causal estimation pipeline on a small toy, linear dataset from `dowhy`

```
python causal_eval/causal_inference_algorithms.py
```
