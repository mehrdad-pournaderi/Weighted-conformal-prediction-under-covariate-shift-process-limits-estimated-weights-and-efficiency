# Code for "Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency"

Mehrdad Pournaderi (Mofid Securities)

Simulation code, data and stored results for the numerical experiments of the paper. The paper itself is not
part of this repository.

The experiments study weighted split conformal prediction under covariate shift: the calibration scores are
reweighted with the likelihood ratio `v = dQ_X/dP_X` of the target and source covariate distributions, and the
quantity of interest is the training-conditional coverage of the resulting prediction set, with known and
with estimated weights. The simulation design (regression model, score, and the four covariate shifts A-D) is
described in the header of [`code/experiments.py`](code/experiments.py); the benchmark on the airfoil data
(run E11) is described in the docstring of `E11` there.

## Layout

```
code/experiments.py     all experiments; the table of runs is at the end of the file
data/                   the airfoil data used by run E11, with its source and license
results/                stored output of every run (JSON)
requirements.txt        Python dependencies
```

## Running

Requirements: Python 3 with NumPy, SciPy and Matplotlib.

```sh
pip install -r requirements.txt
cd code
python3 experiments.py              # every run in the table below
python3 experiments.py E7 E7m8000   # a subset
```

Each run writes one JSON file to `results/`, replacing the stored one. E1, E4 and E5 also write a plot to
`figures/`; the plots are not stored in the repository.

| Run | Content | Output | In the paper | Runtime |
|---|---|---|---|---|
| E1 | functional limit: sup-norm law of the weighted empirical process | `E1_ks.json`, `fig_ks.pdf` | Table 3, Figure 1 | 3 min |
| E2 | simultaneous coverage guarantees over levels | `E2_simultaneous.json` | Table 4 | 1 min |
| E3 | estimated weights, Gaussian mean shift | `E3_estimated_weights.json` | Table 5 | 3 min |
| E4 | heavy-tailed weights (infinite chi-square divergence) | `E4_stable.json`, `fig_stable.pdf` | Figure 3 | 1 min |
| E5 | rates of the Gaussian and bootstrap approximations | `E5_rates.json`, `fig_rates.pdf` | Figure 2 | 5 min |
| E6 | efficiency of the control-variate threshold | `E6_efficiency.json` | Table 10 | 3 min |
| E7 | pairs bootstrap with estimated weights, m = 2000, four configurations | `E7_estboot.json` | Table 7 | 10 min |
| E7m8000 | the same at m = 8000, first three configurations, 1500 replications | `E7_estboot_m8000.json` | Section 9 (text) | 15 min |
| E7b | decomposition of the bootstrap over-rejection | `E7b_decomposition.json` | Table 8 | 20 min |
| E8 | two shifts with the same second moment of the weights | `E8_coupling.json` | Table 9 | 1 min |
| E9 | learned predictor | `E9_learned.json` | Section 9 (text) | 1 min |
| E10 | estimated weights in a bounded parametric model | `E10_estweights_bounded.json` | Table 6 | 15 min |
| E11 | airfoil benchmark: fitted predictor, logistic-regression ratio, naive and pairs bootstrap | `E11_airfoil.json` | focused version, Table 3 | 55 min |

Table and figure numbers refer to the extended version of the paper. The focused version uses E10 (its
Table 2), E11 (Table 3), E2 and E8 (text of its Section 7.3) and E6 (Table 4). Runtimes are approximate, on
one core.

## Reproducing the stored results

The table above is the complete list of runs behind the files in `results/`: the default arguments of every
experiment are the ones used for the paper, and `E7m8000` is the one run with other arguments
(`E7(m=8000, reps=1500, cases=E7_CASES[:3], out_name="E7_estboot_m8000.json", seed=81)`). Each run seeds its
own random generator, so it can be started alone.

The files in `results/` are the output of this code as it stands: all thirteen runs were executed with it
(Python 3.13, NumPy 2.5.3, SciPy 1.18.1, single-threaded OpenBLAS, Linux x86-64), and a rerun on that setup
gives the same files byte for byte.

What exact reproduction depends on:

- **Platform.** The simulations use BLAS matrix products, whose last digits depend on the library and on the
  number of threads. With another NumPy build or a multi-threaded BLAS the output agrees with the stored
  files to numerical precision, not necessarily digit for digit.
- **E7.** Its configurations share one random stream and run in order, so a configuration reproduces a stored
  result only if the same configurations precede it. This is why `E7m8000` runs the first three only.
- **E10.** Each sample size has its own stream (`seed*100000 + m`), so `E10(ms=(500,))` and
  `E10(ms=(2000,))` reproduce their rows separately. A partial rerun is merged into an existing file only if
  that file was produced with the same `reps`, `nboot`, `delta`, `beta` and `seed`, all of which are stored
  in the file; otherwise the run stops before simulating.
- **E3 and E6** use common random numbers across the variants they compare, so that paired differences are
  precise.
- **E11.** Each sample size and each training split has its own stream (`[seed, m, split]`), so
  `E11(ms=(500,))` and `E11(ms=(1000,))` reproduce their rows separately; the stored file was produced that
  way, one sample size at a time, and a second run gave the same file. Within a stream all configurations
  are run on the same calibration samples, in the order of the stored rows.

## Conventions

- **Coverage is computed numerically.** The target score distribution `F_Q` and the clock `G` are tabulated
  by numerical quadrature on a grid of thresholds and interpolated linearly. The coverage `F_Q(tau)` of a
  threshold therefore carries quadrature and interpolation error, but no test-sample Monte Carlo error. In
  E11 the population is finite and `F_Q(tau)` is an exact finite sum.
- **Boundary cases follow the paper.** The weighted quantile is `+inf` at a level above one and when all
  weights are zero, so the prediction set is the whole space. The rank-corrected bootstrap critical value of
  Procedure 1, the `k`-th smallest of `R` draws with `k = ceil((1 - delta)(R + 1))`, is `+inf` when `k > R`.
  Applying these conventions left every stored result unchanged.

## Data

`data/airfoil_self_noise.dat` is the Airfoil Self-Noise data set of the UCI Machine Learning Repository
(Brooks, Pope and Marcolini, 1989, <https://doi.org/10.24432/C5VW2C>), unmodified and distributed under the
Creative Commons Attribution 4.0 International license; see [`data/README.md`](data/README.md).

## Citation

```bibtex
@unpublished{pournaderi2026weighted,
  author = {Pournaderi, Mehrdad},
  title  = {Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency},
  year   = {2026},
  note   = {Preprint}
}
```
