[![arXiv](https://img.shields.io/badge/arXiv-2505.xxxxx-b31b1b.svg)](.)
[![Venue](https://img.shields.io/badge/venue-ICML_2025-darkblue)](.)

# Exogenous Isomorphism for Counterfactual Identifiability

This repository contains the complete code for "[Exogenous Isomorphism for Counterfactual Identifiability](.) [ICML 25 Spotlight]".

## Requirements

Requires `Python >= 3.10`. Install dependencies with

```shell
pip install -r requirements.txt
```

## Reproduce All Experiments

Run experiments:

```shell
bash tmscm_sym_ablation.sh
bash tmscm_sym_exogenous.sh
bash tmscm_er_ablation.sh
```

The results are saved in `experiments` (~22.5GB).

### Produce Figures and Tables

Generate figures and tables from experiment results:

```shell
python graphics.py
```

Then you can find figures and tables in `script/graphics/`. Note that you may need to delete `script/graphics/cache`, `script/graphics/figures` and `script/graphics/tables` first if you want to reproduce new results.

## Citation

Pending...

```bib
```