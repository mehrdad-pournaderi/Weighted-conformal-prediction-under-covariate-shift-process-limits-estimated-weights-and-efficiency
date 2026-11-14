# Code for "Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency"

Mehrdad Pournaderi (Mofid Securities)

Simulation code and stored results for the numerical experiments of the paper. The paper itself is not part of
this repository.

The experiments study weighted split conformal prediction under covariate shift: the calibration scores are
reweighted with the likelihood ratio `v = dQ_X/dP_X` of the target and source covariate distributions, and the
quantity of interest is the training-conditional coverage of the resulting prediction set, with known and
with estimated weights. The simulation design (regression model, score, and the four covariate shifts A-D) is
described in the header of [`code/experiments.py`](code/experiments.py).

## Layout

```
code/experiments.py     all experiments (E1-E10)
results/                stored output of every experiment (JSON)
requirements.txt        Python dependencies
```

## Running

Requirements: Python 3 with NumPy, SciPy and Matplotlib.

```sh
pip install -r requirements.txt
cd code
python3 experiments.py            # everything: E1 ... E10
python3 experiments.py E7 E8      # a subset
```

Each experiment writes its JSON file to `results/`, replacing the stored one. E1, E4 and E5 also write a plot
to `figures/`; the plots are not stored in the repository. Every experiment seeds its own random generator, so
each can be run in isolation and reproduces the stored file exactly.

| Experiment | Content | Output | In the paper | Runtime |
|---|---|---|---|---|
| E1 | functional limit: sup-norm law of the weighted empirical process | `E1_ks.json`, `fig_ks.pdf` | Table 3, Figure 1 | a few min |
| E2 | simultaneous coverage guarantees over levels | `E2_simultaneous.json` | Table 4 | a few min |
| E3 | estimated weights, Gaussian mean shift | `E3_estimated_weights.json` | Table 5 | a few min |
| E4 | heavy-tailed weights (infinite chi-square divergence) | `E4_stable.json`, `fig_stable.pdf` | Figure 3 | a few min |
| E5 | rates of the Gaussian and bootstrap approximations | `E5_rates.json`, `fig_rates.pdf` | Figure 2 | 10 min |
| E6 | efficiency of the control-variate threshold | `E6_efficiency.json` | Table 10 | a few min |
| E7 | pairs bootstrap with estimated weights | `E7_estboot.json`, `E7_estboot_m8000.json` | Table 7 | 10 min |
| E7b | decomposition of the bootstrap over-rejection | `E7b_decomposition.json` | Table 8 | 45 min |
| E8 | two shifts with the same second moment of the weights | `E8_coupling.json` | Table 9 | 1 min |
| E9 | learned predictor | `E9_learned.json` | Section 9 (text) | 1 min |
| E10 | estimated weights in a bounded parametric model | `E10_estweights_bounded.json` | Table 6 | 40 min |

Table and figure numbers refer to the full version of the paper; its journal supplement uses the same numbers
with the prefix S. Runtimes are approximate.

Details that matter for exact reproduction:

- `E7_estboot_m8000.json` comes from
  `E7(m=8000, reps=1500, out_name="E7_estboot_m8000.json", seed=81)`, which is not part of the default run.
- `E7b_decomposition.json` was produced with `reps=1500`.
- E10 seeds its generator per sample size, so `E10(ms=(500,))` and `E10(ms=(2000,))` reproduce the stored
  file and merge into it.
- E3 uses common random numbers across the weight-estimation variants, and E6 computes the oracle and
  control-variate thresholds from the same samples, so that paired differences are precise.

## Citation

```bibtex
@unpublished{pournaderi2026weighted,
  author = {Pournaderi, Mehrdad},
  title  = {Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency},
  year   = {2026},
  note   = {Preprint}
}
```
