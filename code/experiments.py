"""
Numerical illustrations for
  "Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency".

Run:  python3 experiments.py              every run behind the stored files (writes ../results/*.json, ../figures/*.pdf)
      python3 experiments.py E7 E7m8000   a subset; the runs are E1 ... E11, E7b and E7m8000 (see the end of the file)

Model (experiments E1-E10).  Y = 2X + eps, eps | X ~ N(0, (0.2+|X|)^2); fixed predictor mu(x)=2x,
score S = |Y - mu(X)| = |eps|, so F_{S|X=x}(t) = 2 Phi(t/(0.2+|x|)) - 1.
Shifts (likelihood ratio v = dQ_X/dP_X):
  A  bounded:     P_X = U(0,1), Q_X density 4x^3,            v(x) = 4x^3,            B = 4, K^2 = 16/7
  B  tilt:        P_X = N(0,1), Q_X = N(theta,1),            v(x) = exp(theta x - theta^2/2),  K^2 = e^{theta^2}
  C  heavy tail:  P_X = U(0,1), v(x) = c x^{-1/p}, p=1.5,    P(v>t) = c^p t^{-p}: infinite chi-square
  D  bounded, parametric: P_X = Beta(3,1), Q_X = Beta(4,1), v(x) = 4x/3, B = 4/3, K^2 = 16/15, noise scale
     2 - 1.8x instead of 0.2 + |x| (experiment E10: estimated weights, main text)
E11 uses real data instead (../data/airfoil_self_noise.dat): a finite-population design described in its docstring.
Coverage.  F_Q and G are tabulated by numerical quadrature on a grid of thresholds and interpolated linearly, so the
coverage F_Q(tau) of a threshold is computed numerically, without test-sample Monte Carlo error.
Boundary conventions (as in the paper).  A weighted quantile at a level above one, or with all weights zero, is +inf
(the prediction set is the whole space); the rank-corrected bootstrap critical value is +inf when its rank exceeds
the number of draws.
"""
import json, os, time, math
import numpy as np
from scipy import stats, integrate, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20261109)

def sd_eps(x): return 0.2 + np.abs(x)
def FSx(t, x): return 2.0 * stats.norm.cdf(np.asarray(t)[..., None] / sd_eps(np.asarray(x))[None, ...]) - 1.0

class Shift:
    def __init__(self, kind, theta=1.0, p=1.5):
        self.kind = kind
        self.sd = sd_eps          # noise scale of the score model; shift D uses its own
        if kind == "A":
            self.sampP = lambda n: rng.uniform(0, 1, n)
            self.sampQ = lambda n: rng.uniform(0, 1, n) ** 0.25
            self.v = lambda x: 4 * x**3
            self.densQ = lambda x: 4 * x**3
            self.lo, self.hi, self.B, self.K2 = 0.0, 1.0, 4.0, 16 / 7
        elif kind == "B":
            self.theta = theta
            self.sampP = lambda n: rng.normal(0, 1, n)
            self.sampQ = lambda n: rng.normal(theta, 1, n)
            self.v = lambda x: np.exp(theta * x - theta**2 / 2)
            self.densQ = lambda x: stats.norm.pdf(x, theta, 1)
            self.lo, self.hi, self.B, self.K2 = -8.0, 12.0, np.inf, float(np.exp(theta**2))
        elif kind == "D":   # bounded parametric model for the estimated-weights experiment E10
            self.theta_s, self.theta_t = 3.0, 4.0
            self.sd = lambda x: 2.0 - 1.8 * x      # noise large where P_X has its mass, small where Q_X has its mass
            self.sampP = lambda n: rng.uniform(0, 1, n) ** (1 / self.theta_s)
            self.sampQ = lambda n: rng.uniform(0, 1, n) ** (1 / self.theta_t)
            self.v = lambda x: (self.theta_t / self.theta_s) * x ** (self.theta_t - self.theta_s)
            self.densQ = lambda x: self.theta_t * x ** (self.theta_t - 1)
            self.lo, self.hi, self.B, self.K2 = 0.0, 1.0, 4 / 3, 16 / 15
        elif kind == "C":
            self.p = p; c = (p - 1) / p; self.c = c
            self.sampP = lambda n: rng.uniform(0, 1, n)
            self.sampQ = lambda n: rng.uniform(0, 1, n) ** (p / (p - 1))
            self.v = lambda x: c * x ** (-1 / p)
            self.densQ = lambda x: c * x ** (-1 / p)
            self.lo, self.hi, self.B, self.K2 = 0.0, 1.0, np.inf, np.inf
        # tabulate F_Q, G on a grid for fast interpolation
        tg = np.linspace(0, 12, 4001)
        self.tgrid = tg
        self.FQ_tab = np.array([self._FQ(t) for t in tg])
        if np.isfinite(self.K2):
            self.G_tab = np.array([self._G(t) for t in tg])
    def _FQ(self, t):
        if t <= 0: return 0.0
        f = lambda x: self.densQ(x) * (2 * stats.norm.cdf(t / self.sd(x)) - 1)
        return integrate.quad(f, self.lo, self.hi, limit=200)[0]
    def _G(self, t):   # G(t) = E_Q[v 1{S<=t}]
        if t <= 0: return 0.0
        f = lambda x: self.densQ(x) * self.v(x) * (2 * stats.norm.cdf(t / self.sd(x)) - 1)
        return integrate.quad(f, self.lo, self.hi, limit=200)[0]
    def FQ(self, t): return np.interp(t, self.tgrid, self.FQ_tab, left=0.0, right=1.0)
    def qQ(self, beta): return optimize.brentq(lambda t: self.FQ(t) - beta, 1e-9, 12)
    def sample_cal(self, m):
        X = self.sampP(m); S = np.abs(rng.normal(0, 1, m) * self.sd(X)); return X, S, self.v(X)

def wquantile(S, w, beta):
    """self-normalized weighted beta-quantile(s) of S, inf{t: F_hat(t) >= beta}; beta may be an array.
    Conventions of the paper: the value is +inf when no finite t reaches the level, that is, for a level above one
    (in particular an inflated level beta + c/sqrt(m) > 1, or c = +inf) and, with F_hat = 0, when all weights vanish."""
    beta = np.atleast_1d(np.asarray(beta, float))
    o = np.argsort(S); Ss = S[o]; cw = np.cumsum(np.asarray(w, dtype=float)[o])   # float accumulation: integer weights are valid
    if not cw[-1] > 0: return np.full(beta.shape, np.inf)
    cw /= cw[-1]
    k = np.searchsorted(cw, beta, side="left")
    return np.where(k < len(Ss), Ss[np.minimum(k, len(Ss) - 1)], np.inf)

def rank_quantile(draws, delta):
    """critical value of Procedure 1: the k-th smallest of the R draws, k = ceil((1-delta)(R+1)); +inf if k > R."""
    R = len(draws); k = int(np.ceil((1 - delta) * (R + 1)))
    return float(np.sort(draws)[k - 1]) if k <= R else np.inf

def save(name, obj):
    with open(os.path.join(RES, name), "w") as f: json.dump(obj, f, indent=1)

# ----------------------------------------------------------------------------------------------
# E1.  Weighted Kolmogorov-Smirnov statistic vs. the law of sup|G_Q| from the Brownian representation
# ----------------------------------------------------------------------------------------------
def E1(reps=3000, npaths=20000):
    global rng; rng = np.random.default_rng(1)      # each experiment is seeded on its own and reproducible in isolation
    out = {}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, kind in zip(axes, ["A", "B"]):
        sh = Shift(kind)
        # limit:  sup_t | W(G(t)) - F_Q(t) W(K^2) |  on the tabulated grid
        G = sh.G_tab; F = sh.FQ_tab; K2 = sh.K2
        dG = np.diff(np.concatenate([[0.0], G]))
        sup_lim = np.empty(npaths)
        bs = 2000
        for a in range(0, npaths, bs):
            inc = rng.normal(0, 1, (bs, len(dG))) * np.sqrt(dG)[None, :]
            WG = np.cumsum(inc, axis=1)
            WK = WG[:, -1] + rng.normal(0, 1, bs) * np.sqrt(max(K2 - G[-1], 0.0))
            sup_lim[a:a + bs] = np.max(np.abs(WG - F[None, :] * WK[:, None]), axis=1)
        res = {"K2": K2, "B": sh.B, "reps": reps, "npaths": npaths, "limit_quantiles": {}}
        qs = [0.5, 0.9, 0.95, 0.99]
        for q in qs: res["limit_quantiles"][str(q)] = float(np.quantile(sup_lim, q))
        for m in [500, 2000]:
            ks = np.empty(reps)
            for r in range(reps):
                X, S, v = sh.sample_cal(m)
                o = np.argsort(S); Ss = S[o]; cw = np.cumsum(v[o]) / v.sum()
                Fq = sh.FQ(Ss)
                # sup over the step function: check at jumps (right limit) and just before
                ks[r] = np.sqrt(m) * max(np.max(np.abs(cw - Fq)), np.max(np.abs(np.concatenate([[0.0], cw[:-1]]) - Fq)))
            res[f"m={m}"] = {str(q): float(np.quantile(ks, q)) for q in qs}
            res[f"m={m}"]["mean"] = float(ks.mean())
            if m == 2000:
                pr = np.linspace(0.005, 0.995, 199)
                ax.plot(np.quantile(sup_lim, pr), np.quantile(ks, pr), ".", ms=3)
        res["limit_mean"] = float(sup_lim.mean())
        lim = [0, max(np.quantile(sup_lim, 0.995), 1) * 1.05]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.set_xlabel(r"quantiles of $\|\mathbb{G}_Q\|_\infty$ (limit)")
        ax.set_ylabel(r"quantiles of $\sqrt{m}\,\|\hat F_m-F_Q\|_\infty$, $m=2000$")
        ax.set_title(f"shift {kind}:  $K^2$={K2:.2f}" + (f", $B$={sh.B:g}" if np.isfinite(sh.B) else ", $B=\\infty$"))
        out[kind] = res
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ks.pdf")); plt.close(fig)
    save("E1_ks.json", out); return out

# ----------------------------------------------------------------------------------------------
# E2.  Simultaneous (in the level) training-conditional guarantees: joint failure over several levels
# ----------------------------------------------------------------------------------------------
def E2(m=2000, delta=0.1, reps=1500, nboot=400):
    global rng; rng = np.random.default_rng(2)
    sh = Shift("A")
    levels_fin = np.array([0.80, 0.85, 0.90, 0.95])
    levels_int = np.linspace(0.80, 0.95, 61)         # interval of levels, handled on a grid
    z1 = stats.norm.ppf(1 - delta); zbonf = stats.norm.ppf(1 - delta / len(levels_fin))
    rec = {"finite": {k: [] for k in ["none", "pointwise", "bonferroni", "joint_boot"]},
           "interval": {k: [] for k in ["none", "pointwise", "joint_boot"]},
           "inflation": {"pointwise": [], "bonferroni": [], "joint_boot_fin": [], "pointwise_int": [], "joint_boot_int": []}}
    for r in range(reps):
        X, S, v = sh.sample_cal(m); vbar = v.mean()
        for name, levels in [("finite", levels_fin), ("interval", levels_int)]:
            tau = wquantile(S, v, levels)
            ind = (S[:, None] <= tau[None, :]).astype(float)                      # m x L
            Fhat = (v[:, None] * ind).sum(0) / v.sum()
            H = v[:, None] * (ind - Fhat[None, :])                                 # m x L, centered at Fhat
            sig = np.sqrt((H**2).mean(0)) / vbar                                   # plug-in sd of Z_m(tau(beta))
            # multiplier bootstrap of the one-sided sup of the coverage process
            xi = rng.normal(0, 1, (nboot, m))
            Zb = (xi @ H) / (np.sqrt(m) * vbar)                                    # nboot x L
            chat = rank_quantile(Zb.max(1), delta)                                 # critical value of Procedure 1
            def fail(infl):   # any level under-covers?  P_e(tau(beta+infl)) > 1-beta  <=>  F_Q(tau(beta+infl)) < beta
                t2 = wquantile(S, v, levels + infl)                                # +inf where the level exceeds one
                return bool(np.any(sh.FQ(t2) < levels))
            rec[name]["none"].append(fail(0.0))
            rec[name]["pointwise"].append(fail(z1 * sig / np.sqrt(m)))
            rec[name]["joint_boot"].append(fail(chat / np.sqrt(m)))
            if name == "finite":
                rec[name]["bonferroni"].append(fail(zbonf * sig / np.sqrt(m)))
                rec["inflation"]["pointwise"].append(float(np.mean(z1 * sig / np.sqrt(m))))
                rec["inflation"]["bonferroni"].append(float(np.mean(zbonf * sig / np.sqrt(m))))
                rec["inflation"]["joint_boot_fin"].append(float(chat / np.sqrt(m)))
            else:
                rec["inflation"]["pointwise_int"].append(float(np.mean(z1 * sig / np.sqrt(m))))
                rec["inflation"]["joint_boot_int"].append(float(chat / np.sqrt(m)))
    out = {"m": m, "delta": delta, "reps": reps, "nboot": nboot,
           "levels_finite": levels_fin.tolist(), "interval": [0.80, 0.95],
           "joint_failure": {name: {k: float(np.mean(vals)) for k, vals in d.items()} for name, d in rec.items() if name != "inflation"},
           "mean_inflation": {k: float(np.mean(vals)) for k, vals in rec["inflation"].items()}}
    save("E2_simultaneous.json", out); return out

# ----------------------------------------------------------------------------------------------
# E3.  Estimated likelihood ratio (Gaussian mean shift, theta_hat = mean(X_target) - mean(X_source))
# ----------------------------------------------------------------------------------------------
def E3(m=2000, alpha=0.2, reps=20000):
    """Common random numbers: every variant uses the same calibration sample and the same auxiliary draws,
    so that paired differences from the oracle are estimated far more precisely than the variances themselves."""
    global rng; rng = np.random.default_rng(3)
    sh = Shift("B"); theta = sh.theta; beta = 1 - alpha
    q = sh.qQ(beta)
    # sigma_alpha^2 = E_P[v^2 (1{S<=q} - beta)^2]  and  gamma(q) = Cov_Q(X, F_{S|X}(q))
    f1 = lambda x: stats.norm.pdf(x) * sh.v(x)**2 * ((1 - beta)**2 * (2 * stats.norm.cdf(q / sd_eps(x)) - 1) + beta**2 * (2 - 2 * stats.norm.cdf(q / sd_eps(x))))
    sigma2 = integrate.quad(f1, -8, 12, limit=200)[0]
    f2 = lambda x: sh.densQ(x) * (x - theta) * (2 * stats.norm.cdf(q / sd_eps(x)) - 1)
    gamma = integrate.quad(f2, -8, 12, limit=200)[0]
    # variant -> (lambda_t, source) ; lambda_t = m / N_t ; source "cal" = calibration covariates, "indep" = independent sample of size m
    variants = {"oracle": (None, "none"), "cal_Nt=4m": (0.25, "cal"), "cal_Nt=m": (1.0, "cal"), "indep_Nt=4m": (0.25, "indep"),
                "indep_Nt=m": (1.0, "indep"), "cal_Nt=m/4": (4.0, "cal")}
    def pred(lam_t, src):
        if src == "none": return sigma2
        if src == "indep": return sigma2 + gamma**2 * (lam_t + 1.0)
        return sigma2 + gamma**2 * (lam_t - 1.0)
    Z = {k: np.empty(reps) for k in variants}
    for r in range(reps):
        X, S, v = sh.sample_cal(m)
        Xt_all = sh.sampQ(4 * m)                 # nested target samples: first N_t points
        Xs = sh.sampP(m)                          # independent source sample
        for k, (lam_t, src) in variants.items():
            if src == "none":
                w = v
            else:
                Nt = int(round(m / lam_t)); Xt = Xt_all[:Nt]
                th = Xt.mean() - (Xs.mean() if src == "indep" else X.mean())
                w = np.exp(th * X)               # self-normalized: the normalizing constant cancels
            tau = wquantile(S, w, beta)[0]
            Z[k][r] = np.sqrt(m) * ((1 - sh.FQ(tau)) - alpha)
    out = {"m": m, "alpha": alpha, "reps": reps, "sigma2_alpha": sigma2, "gamma_q": gamma, "q": q, "variants": {}}
    Zo = Z["oracle"]
    for k, (lam_t, src) in variants.items():
        D = Z[k]**2 - Zo**2                       # paired contribution to the variance difference
        out["variants"][k] = {"pred_var": float(pred(lam_t, src)), "emp_var": float(Z[k].var()), "emp_mean": float(Z[k].mean()),
                              "pred_diff_vs_oracle": float(pred(lam_t, src) - sigma2),
                              "emp_diff_vs_oracle": float(Z[k].var() - Zo.var()),
                              "se_diff": float(D.std(ddof=1) / np.sqrt(reps)),
                              "ks_p_centered": float(stats.kstest((Z[k] - Z[k].mean()) / np.sqrt(pred(lam_t, src)), "norm").pvalue)}
    out["se_var_oracle"] = float((Zo**2).std(ddof=1) / np.sqrt(reps))
    save("E3_estimated_weights.json", out); return out

# ----------------------------------------------------------------------------------------------
# E4.  Heavy-tailed likelihood ratio (infinite chi-square): rate m^{1-1/p} and a stable limit
# ----------------------------------------------------------------------------------------------
def E4(beta=0.8, reps=3000):
    global rng; rng = np.random.default_rng(4)
    sh = Shift("C"); p = sh.p; c = sh.c
    q = sh.qQ(beta)
    pi_beta = 2 * stats.norm.cdf(q / 0.2) - 1          # Pi = law of 0.2|N(0,1)|  (scores in the far tail of the shift, x -> 0)
    cp, cm = pi_beta * (1 - beta)**p, (1 - pi_beta) * beta**p
    skew = (cp - cm) / (cp + cm)
    # S_p(scale, skew, 0) in the S1 parametrization: scale^p = Gamma(1-p) cos(pi p/2) (c_+ + c_-)  (positive for 1<p<2)
    scale = (math.gamma(1 - p) * np.cos(np.pi * p / 2) * (cp + cm)) ** (1 / p)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ms = [250, 1000, 4000, 16000]
    iqr = {}; store = {}
    for m in ms:
        D = np.empty(reps)
        for r in range(reps):
            X, S, v = sh.sample_cal(m)
            tau = wquantile(S, v, beta)[0]
            D[r] = (1 - sh.FQ(tau)) - (1 - beta)
        iqr[m] = float(np.quantile(D, 0.75) - np.quantile(D, 0.25)); store[m] = D
    lm = np.log(ms); li = np.log([iqr[m] for m in ms])
    slope = float(np.polyfit(lm, li, 1)[0])
    # stable comparison at the largest m
    m = ms[-1]; a_m = c * m ** (1 / p)
    Zs = (m / a_m) * store[m]
    pr = np.linspace(0.01, 0.99, 99)
    st = stats.levy_stable(p, skew, loc=0.0, scale=scale)
    qs_stable = st.ppf(pr); qs_emp = np.quantile(Zs, pr)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    ax = axes[0]
    ax.loglog(ms, [iqr[m] for m in ms], "o-", label=f"IQR of $P_e-\\alpha$ (slope {slope:.2f})")
    ref = iqr[ms[0]] * (np.array(ms) / ms[0]) ** (-(1 - 1 / p))
    ax.loglog(ms, ref, "k--", lw=0.8, label=f"slope $-(1-1/p)=-{1-1/p:.2f}$")
    ref2 = iqr[ms[0]] * (np.array(ms) / ms[0]) ** (-0.5)
    ax.loglog(ms, ref2, "k:", lw=0.8, label="slope $-1/2$")
    ax.set_xlabel("$m$"); ax.set_ylabel("IQR"); ax.legend(fontsize=7); ax.set_title(f"shift C, $p={p}$, $\\beta={beta}$")
    ax = axes[1]
    ax.plot(qs_stable, qs_emp, ".", ms=3, label="vs. predicted stable law")
    gq = stats.norm.ppf(pr, loc=Zs.mean(), scale=Zs.std())
    ax.plot(gq, qs_emp, "x", ms=3, label="vs. fitted Gaussian")
    lim = [min(qs_emp.min(), qs_stable.min()), max(qs_emp.max(), qs_stable.max())]
    ax.plot(lim, lim, "k-", lw=0.8); ax.set_xlabel("theoretical quantiles"); ax.set_ylabel(f"empirical quantiles of $(m/a_m)(P_e-\\alpha)$, $m={m}$")
    ax.legend(fontsize=7); ax.set_title(f"$\\pi_\\beta$={pi_beta:.2f}, skewness={skew:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_stable.pdf")); plt.close(fig)
    out = {"p": p, "beta": beta, "reps": reps, "q": q, "pi_beta": pi_beta, "skewness": skew, "scale": float(scale),
           "iqr": {str(k): v for k, v in iqr.items()}, "loglog_slope": slope, "predicted_slope": -(1 - 1 / p),
           "m_for_qq": m, "a_m": a_m, "mean_Z": float(Zs.mean()), "emp_quantiles_Z": {str(k): float(np.quantile(Zs, k)) for k in [0.05, 0.25, 0.5, 0.75, 0.95]},
           "stable_quantiles": {str(k): float(st.ppf(k)) for k in [0.05, 0.25, 0.5, 0.75, 0.95]}}
    save("E4_stable.json", out); return out


# ----------------------------------------------------------------------------------------------
# E5.  Rates: Kolmogorov distance to the Gaussian limit for a single level and for the grid maximum
# ----------------------------------------------------------------------------------------------
def E5(ms=(50, 100, 200, 400, 800, 1600, 3200, 6400), reps=200000, levels=(0.5, 0.6, 0.7, 0.8, 0.9), nboot=400, reps_boot=10000, delta=0.1):
    global rng; rng = np.random.default_rng(6)
    sh = Shift("A"); K2 = sh.K2
    b = np.array(levels); L = len(b)
    q = np.array([sh.qQ(x) for x in b]); Gt = np.interp(q, sh.tgrid, sh.G_tab)
    sig2 = (1 - b)**2 * Gt + b**2 * (K2 - Gt); sig = np.sqrt(sig2)
    # covariance of (L(b_j)) and of the standardized vector
    C = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            bm = min(b[i], b[j]); C[i, j] = np.interp(sh.qQ(bm), sh.tgrid, sh.G_tab) - b[i] * Gt[j] - b[j] * Gt[i] + K2 * b[i] * b[j]
    Cs = C / np.outer(sig, sig)
    # law of the Gaussian maximum: many draws
    Lch = np.linalg.cholesky(Cs + 1e-12 * np.eye(L))
    G = (rng.normal(0, 1, (2_000_000, L)) @ Lch.T)
    Tg = np.sort(G.max(1)); Tg_abs = np.sort(np.abs(G).max(1))
    def ks_to(sample, ref_sorted):
        s = np.sort(sample); n = len(s)
        Fref = np.searchsorted(ref_sorted, s, side="right") / len(ref_sorted)
        return float(max(np.max(np.abs(np.arange(1, n + 1) / n - Fref)), np.max(np.abs(np.arange(0, n) / n - Fref))))
    out = {"levels": levels, "sigma": sig.tolist(), "reps": reps, "ms": list(ms), "single": {}, "grid_onesided": {}, "grid_twosided": {}, "boot": {}}
    for m in ms:
        bs = max(1, int(4e6 // m)); Tm = np.empty(reps); Tma = np.empty(reps); Z1 = np.empty(reps)
        done = 0
        while done < reps:
            B = min(bs, reps - done)
            X = sh.sampP(B * m).reshape(B, m); S = np.abs(rng.normal(0, 1, (B, m)) * sd_eps(X)); v = sh.v(X)
            o = np.argsort(S, 1); Ss = np.take_along_axis(S, o, 1); cw = np.cumsum(np.take_along_axis(v, o, 1), 1); cw /= cw[:, -1:]
            # weighted quantiles at the levels: first index with cw >= b
            idx = np.stack([(cw >= bj).argmax(1) for bj in b], 1)               # B x L
            tau = np.take_along_axis(Ss, idx, 1)
            Cm = b[None, :] - sh.FQ(tau)                                        # B x L
            Zs = np.sqrt(m) * Cm / sig[None, :]
            Tm[done:done + B] = Zs.max(1); Tma[done:done + B] = np.abs(Zs).max(1); Z1[done:done + B] = Zs[:, 3]   # level 0.8
            done += B
        out["single"][str(m)] = float(stats.kstest(Z1, "norm").statistic)
        out["grid_onesided"][str(m)] = ks_to(Tm, Tg); out["grid_twosided"][str(m)] = ks_to(Tma, Tg_abs)
    # multiplier bootstrap calibration of the one-sided grid max (studentized by sigma_hat), smaller reps
    for m in (100, 400, 1600):
        fails = 0
        for r in range(reps_boot):
            X, S, v = sh.sample_cal(m); vbar = v.mean()
            tau = wquantile(S, v, b); ind = (S[:, None] <= tau[None, :]).astype(float)
            Fhat = (v[:, None] * ind).sum(0) / v.sum(); H = v[:, None] * (ind - Fhat[None, :])
            sighat = np.sqrt((v[:, None]**2 * (ind - b[None, :])**2).mean(0)) / vbar
            xi = rng.normal(0, 1, (nboot, m)); Zb = (xi @ H) / (np.sqrt(m) * vbar) / sighat[None, :]
            chat = np.quantile(Zb.max(1), 1 - delta)
            That = np.max(np.sqrt(m) * (b - sh.FQ(tau)) / sighat)
            fails += That > chat
        out["boot"][str(m)] = {"delta": delta, "rejection": fails / reps_boot, "reps": reps_boot, "nboot": nboot}
    # log-log slopes
    lm = np.log(np.array(ms, float))
    for key in ["single", "grid_onesided", "grid_twosided"]:
        y = np.log(np.array([out[key][str(m)] for m in ms]))
        out[key + "_slope"] = float(np.polyfit(lm, y, 1)[0])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.6))
    for key, lab, mk in [("single", r"single level $\beta=0.8$", "o"), ("grid_onesided", r"one-sided max over 5 levels", "s"), ("grid_twosided", r"two-sided max over 5 levels", "^")]:
        ax.loglog(ms, [out[key][str(m)] for m in ms], mk + "-", ms=4, label=lab + f" (slope {out[key+'_slope']:.2f})")
    ax.loglog(ms, 0.5 * np.array(ms, float)**-0.5, "k:", lw=0.8, label=r"$\propto m^{-1/2}$")
    ax.loglog(ms, 0.3 * np.array(ms, float)**-0.25, "k--", lw=0.8, label=r"$\propto m^{-1/4}$")
    ax.set_xlabel("$m$"); ax.set_ylabel("Kolmogorov distance to the Gaussian limit"); ax.legend(fontsize=6.5); ax.set_title("shift A")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rates.pdf")); plt.close(fig)
    save("E5_rates.json", out); return out

# ----------------------------------------------------------------------------------------------
# E6.  Efficiency: oracle weighted threshold vs the control-variate threshold; the three bounds
# ----------------------------------------------------------------------------------------------
def E6(m=2000, reps=20000, betas=(0.8, 0.9), kinds=("A", "B")):
    global rng; rng = np.random.default_rng(7)
    out = {"m": m, "reps": reps}
    for kind in kinds:
        sh = Shift(kind); K2 = sh.K2; out[kind] = {}
        for beta in betas:
            q = sh.qQ(beta); Gt = float(np.interp(q, sh.tgrid, sh.G_tab))
            sigma2 = (1 - beta)**2 * Gt + beta**2 * (K2 - Gt)
            sigma2_eff = sigma2 - (beta * K2 - Gt)**2 / (K2 - 1)
            mt = lambda x: 2 * stats.norm.cdf(q / sd_eps(x)) - 1                      # m_t(x) = P(S<=q | X=x)
            V2 = integrate.quad(lambda x: sh.densQ(x) * sh.v(x) * mt(x) * (1 - mt(x)), sh.lo, sh.hi, limit=200)[0]   # E_P[v^2 m(1-m)] = E_Q[v m (1-m)]
            varQm = integrate.quad(lambda x: sh.densQ(x) * (mt(x) - beta)**2, sh.lo, sh.hi, limit=200)[0]
            V3 = {str(r_): V2 + r_ * varQm for r_ in (1.0, 0.25)}
            Zo = np.empty(reps); Zc = np.empty(reps); Zu = np.empty(reps)
            for r in range(reps):
                X, S, v = sh.sample_cal(m)
                o = np.argsort(S); Ss = S[o]; vs = v[o]
                cw = np.cumsum(vs); Fh = cw / cw[-1]                                   # self-normalized
                tau_o = Ss[np.searchsorted(Fh, beta)]
                Fu = cw / m                                                             # unnormalized
                k = np.searchsorted(Fu, beta); tau_u = Ss[k] if k < m else np.inf
                Gm = np.cumsum(vs**2) / m; K2m = (v**2).mean(); vbar = v.mean()
                ch = (Gm - Fh) / (K2m - 1)
                Ft = Fu - ch * (vbar - 1)
                idx = np.nonzero(Ft >= beta)[0]; tau_c = Ss[idx[0]] if len(idx) else np.inf
                Zo[r] = np.sqrt(m) * (beta - sh.FQ(tau_o)); Zc[r] = np.sqrt(m) * (beta - sh.FQ(tau_c)); Zu[r] = np.sqrt(m) * (beta - sh.FQ(tau_u))
            out[kind][str(beta)] = {"K2": K2, "Gtilde": Gt, "sigma2_oracle": sigma2, "sigma2_eff": sigma2_eff,
                                    "V2_known_marginals": V2, "V3_unknown_ratio": V3, "VarQ_m": varQm,
                                    "emp_var_oracle": float(Zo.var(ddof=1)), "emp_var_cv": float(Zc.var(ddof=1)),
                                    "emp_var_unnormalized": float(Zu.var(ddof=1)),
                                    "pred_var_unnormalized": Gt - beta**2,
                                    "emp_mean_oracle": float(Zo.mean()), "emp_mean_cv": float(Zc.mean()),
                                    "paired_var_diff": float(np.var(Zo, ddof=1) - np.var(Zc, ddof=1)),
                                    "se_var_oracle": float(np.std(Zo**2, ddof=1) / np.sqrt(reps))}
    save("E6_efficiency.json", out); return out


# ----------------------------------------------------------------------------------------------
# E7.  Feasible (pairs) bootstrap with estimated weights vs the naive bootstrap with the weights held fixed
# ----------------------------------------------------------------------------------------------
E7_CASES = ("indep_Nt=m", "cal_Nt=4m", "cal_Nt=m/4", "cal_Nt=m/16")
def E7(m=2000, reps=3000, nboot=400, delta=0.1, levels=(0.8, 0.85, 0.9, 0.95), out_name="E7_estboot.json", seed=8, cases=E7_CASES):
    global rng; rng = np.random.default_rng(seed)
    sh = Shift("B"); theta = sh.theta; b = np.array(levels); j0 = 0          # single level = levels[0] = 0.8
    q = np.array([sh.qQ(x) for x in b]); Gt = np.interp(q, sh.tgrid, sh.G_tab)
    sigma2 = (1 - b)**2 * Gt + b**2 * (sh.K2 - Gt)
    gam = np.array([integrate.quad(lambda x: sh.densQ(x) * (x - theta) * (2 * stats.norm.cdf(qq / sd_eps(x)) - 1), -8, 12, limit=200)[0] for qq in q])
    allcases = {"indep_Nt=m": ("indep", 1.0, 1.0), "cal_Nt=4m": ("cal", 0.25, None), "cal_Nt=m/4": ("cal", 4.0, None),
                "cal_Nt=m/16": ("cal", 16.0, None)}
    # The configurations share one random stream and run in the order given, so a configuration reproduces a stored
    # result only if the same configurations precede it (E7_estboot_m8000.json was produced with the first three).
    out = {"m": m, "reps": reps, "nboot": nboot, "delta": delta, "levels": levels, "sigma2": sigma2.tolist(), "gamma": gam.tolist(), "cases": {}}
    for name in cases:
        src, lam_t, lam_s = allcases[name]
        Nt = int(round(m / lam_t)); Ns = int(round(m / lam_s)) if lam_s else None
        tau2 = sigma2 + gam**2 * ((lam_t + lam_s) if src == "indep" else (lam_t - 1.0))
        rec = {k: np.zeros(reps, bool) for k in ["naive_single", "feas_single", "naive_joint", "feas_joint"]}
        sd_naive = np.empty(reps); sd_feas = np.empty(reps); Zc = np.empty(reps)
        for r in range(reps):
            X, S, v = sh.sample_cal(m); Xt = sh.sampQ(Nt)
            Xs = sh.sampP(Ns) if src == "indep" else X
            th = Xt.mean() - Xs.mean(); vh = np.exp(th * X)
            tau = wquantile(S, vh, b)                                          # estimated-weight thresholds
            ind = (S[:, None] <= tau[None, :]).astype(float); Fc = (vh[:, None] * ind).sum(0) / vh.sum()
            Zc[r] = np.sqrt(m) * (b[j0] - sh.FQ(tau[j0]))
            # naive multiplier bootstrap, weights fixed
            xi = rng.normal(0, 1, (nboot, m)); H = vh[:, None] * (ind - Fc[None, :])
            Zn = (xi @ H) / (np.sqrt(m) * vh.mean())
            cn1 = np.quantile(Zn[:, j0], 1 - delta); cnJ = np.quantile(Zn.max(1), 1 - delta); sd_naive[r] = Zn[:, j0].std()
            # pairs bootstrap: resample calibration pairs and the auxiliary samples, recompute theta and the weights
            idx = rng.integers(0, m, (nboot, m)); Xb = X[idx]; Sb = S[idx]
            Xtb = Xt[rng.integers(0, Nt, (nboot, Nt))].mean(1)
            Xsb = Xs[rng.integers(0, Ns, (nboot, Ns))].mean(1) if src == "indep" else Xb.mean(1)
            thb = Xtb - Xsb; vb = np.exp(thb[:, None] * Xb)                   # nboot x m
            Fb = np.stack([(vb * (Sb <= t)).sum(1) / vb.sum(1) for t in tau], 1)   # nboot x L
            Lam = np.sqrt(m) * (Fb - Fc[None, :])
            cf1 = np.quantile(Lam[:, j0], 1 - delta); cfJ = np.quantile(Lam.max(1), 1 - delta); sd_feas[r] = Lam[:, j0].std()
            def fail(infl, single):
                lev = b + infl if not single else np.array([b[j0] + infl])       # +inf threshold where the level exceeds one
                t2 = wquantile(S, vh, lev); target = b if not single else b[[j0]]
                return bool(np.any(sh.FQ(t2) < target))
            rec["naive_single"][r] = fail(cn1 / np.sqrt(m), True); rec["feas_single"][r] = fail(cf1 / np.sqrt(m), True)
            rec["naive_joint"][r] = fail(cnJ / np.sqrt(m), False); rec["feas_joint"][r] = fail(cfJ / np.sqrt(m), False)
        z1 = stats.norm.ppf(1 - delta)
        out["cases"][name] = {"Nt": Nt, "Ns": Ns, "tau2_pred_single": float(tau2[j0]), "sigma2_single": float(sigma2[j0]),
                              "limit_failure_naive_single": float(1 - stats.norm.cdf(z1 * np.sqrt(sigma2[j0] / tau2[j0]))),
                              "emp_var_single": float(Zc.var(ddof=1)),
                              "mean_boot_var_naive": float((sd_naive**2).mean()), "mean_boot_var_feas": float((sd_feas**2).mean()),
                              "failure": {k: float(vals.mean()) for k, vals in rec.items()}}
    save(out_name, out); return out

# ----------------------------------------------------------------------------------------------
# E7b. Where does the over-rejection of the inflated sets with estimated weights come from?  Single level.
#      Failure rates of the set run at beta + c/sqrt(m) for c = z tau (fixed population critical value, the
#      asymptotic target), c = z sigma (the naive asymptotic target), and bootstrap critical values with R draws
#      (naive multiplier with fixed weights, pairs), using np.quantile and the rank-corrected order statistic.
#      By quantile duality the failure event is {sqrt(m)(F_check(q-) - beta) >= c}; Zq records that statistic.
# ----------------------------------------------------------------------------------------------
def E7b(m=2000, reps=1500, nboots=(400, 4000), delta=0.1, beta=0.8, cases=("indep_Nt=m", "cal_Nt=m/16"), seed=82,
        out_name="E7b_decomposition.json"):
    global rng; rng = np.random.default_rng(seed)
    sh = Shift("B"); theta = sh.theta; q = sh.qQ(beta); Gt = float(np.interp(q, sh.tgrid, sh.G_tab))
    sigma2 = (1 - beta)**2 * Gt + beta**2 * (sh.K2 - Gt)
    gam = integrate.quad(lambda x: sh.densQ(x) * (x - theta) * (2 * stats.norm.cdf(q / sd_eps(x)) - 1), -8, 12, limit=200)[0]
    allcases = {"indep_Nt=m": ("indep", 1.0, 1.0), "cal_Nt=4m": ("cal", 0.25, None), "cal_Nt=m/4": ("cal", 4.0, None), "cal_Nt=m/16": ("cal", 16.0, None)}
    z1 = stats.norm.ppf(1 - delta)
    out = {"m": m, "reps": reps, "nboots": list(nboots), "delta": delta, "beta": beta, "sigma2": sigma2, "gamma": gam, "cases": {}}
    for name in cases:
        src, lam_t, lam_s = allcases[name]
        Nt = int(round(m / lam_t)); Ns = int(round(m / lam_s)) if lam_s else None
        tau2 = sigma2 + gam**2 * ((lam_t + lam_s) if src == "indep" else (lam_t - 1.0))
        keys = ["fixed_tau", "fixed_sigma"] + [f"{kind}_R{R}{suf}" for R in nboots for kind in ("naive", "pairs") for suf in ("", "_rank")]
        fails = {k: np.zeros(reps, bool) for k in keys}; Zq = np.empty(reps); Zc = np.empty(reps)
        cvals = {k: np.empty(reps) for k in keys}
        for r in range(reps):
            X, S, v = sh.sample_cal(m); Xt = sh.sampQ(Nt); Xs = sh.sampP(Ns) if src == "indep" else X
            th = Xt.mean() - Xs.mean(); vh = np.exp(th * X)
            tau = wquantile(S, vh, beta)[0]; Fc = (vh * (S <= tau)).sum() / vh.sum()
            Zc[r] = np.sqrt(m) * (beta - sh.FQ(tau)); Zq[r] = np.sqrt(m) * ((vh * (S < q)).sum() / vh.sum() - beta)
            ind = (S <= tau).astype(float); H = vh * (ind - Fc)
            def fail(c):
                lev = beta + c / np.sqrt(m); return bool(sh.FQ(wquantile(S, vh, lev)[0]) < beta)   # +inf threshold if lev > 1
            cvals["fixed_tau"][r] = z1 * np.sqrt(tau2); cvals["fixed_sigma"][r] = z1 * np.sqrt(sigma2)
            for R in nboots:
                xi = rng.normal(0, 1, (R, m)); Zn = (xi @ H) / (np.sqrt(m) * vh.mean())
                idx = rng.integers(0, m, (R, m)); Xb = X[idx]; Sb = S[idx]
                Xtb = Xt[rng.integers(0, Nt, (R, Nt))].mean(1)
                Xsb = Xs[rng.integers(0, Ns, (R, Ns))].mean(1) if src == "indep" else Xb.mean(1)
                vb = np.exp((Xtb - Xsb)[:, None] * Xb); Lam = np.sqrt(m) * ((vb * (Sb <= tau)).sum(1) / vb.sum(1) - Fc)
                for kind, draws in (("naive", Zn), ("pairs", Lam)):
                    cvals[f"{kind}_R{R}"][r] = np.quantile(draws, 1 - delta)
                    cvals[f"{kind}_R{R}_rank"][r] = rank_quantile(draws, delta)
            for k in keys: fails[k][r] = fail(cvals[k][r])
        with np.errstate(invalid="ignore"):   # an infinite critical value (rank above R) has mean +inf and no sd or correlation
            rec = {"Nt": Nt, "Ns": Ns, "tau2_pred": float(tau2), "emp_var_Zc": float(Zc.var(ddof=1)), "emp_var_Zq": float(Zq.var(ddof=1)),
                   "limit_failure_naive": float(1 - stats.norm.cdf(z1 * np.sqrt(sigma2 / tau2))),
                   "failure": {k: float(f.mean()) for k, f in fails.items()},
                   "mean_critical_value": {k: float(c.mean()) for k, c in cvals.items()},
                   "sd_critical_value": {k: float(c.std(ddof=1)) for k, c in cvals.items()},
                   "corr_critical_value_Zq": {k: float(np.corrcoef(c, Zq)[0, 1]) for k, c in cvals.items() if c.std() > 0}}
        out["cases"][name] = rec
    save(out_name, out); return out

# ----------------------------------------------------------------------------------------------
# E8.  Same K^2, different coupling between weights and scores: the clock G, not only m/K^2, sets the variance
# ----------------------------------------------------------------------------------------------
def E8(m=2000, reps=20000, betas=(0.8, 0.9)):
    global rng; rng = np.random.default_rng(9)
    K2 = 16 / 7; Phi = stats.norm.cdf
    # three shifts with P_X uniform and v in {4x^3 (increasing), 4(1-x)^3 (decreasing), 4x_2^3 with x_2 independent of the score}
    def make(kind):
        if kind == "inc":
            v = lambda x: 4 * x**3; dens = lambda x: 4 * x**3; sc = lambda x: x
        elif kind == "dec":
            v = lambda x: 4 * (1 - x)**3; dens = lambda x: 4 * (1 - x)**3; sc = lambda x: x
        FQ = lambda t: integrate.quad(lambda x: dens(x) * (2 * Phi(t / sd_eps(x)) - 1), 0, 1, limit=200)[0]
        G = lambda t: integrate.quad(lambda x: dens(x) * v(x) * (2 * Phi(t / sd_eps(x)) - 1), 0, 1, limit=200)[0]
        return v, dens, FQ, G
    FP = lambda t: integrate.quad(lambda x: (2 * Phi(t / sd_eps(x)) - 1), 0, 1, limit=200)[0]
    tg = np.linspace(0, 12, 2001)
    out = {"m": m, "reps": reps, "K2": K2, "shifts": {}}
    for kind in ["inc", "dec", "ind"]:
        if kind != "ind":
            v, dens, FQ, G = make(kind)
            FQ_tab = np.array([FQ(t) for t in tg]); G_tab = np.array([G(t) for t in tg])
        else:
            FQ_tab = np.array([FP(t) for t in tg]); G_tab = K2 * FQ_tab
        res = {}
        for beta in betas:
            q = float(np.interp(beta, FQ_tab, tg)); Gt = float(np.interp(q, tg, G_tab))
            sigma2 = (1 - beta)**2 * Gt + beta**2 * (K2 - Gt); kish = K2 * beta * (1 - beta)
            Z = np.empty(reps)
            for r in range(reps):
                if kind == "ind":
                    X1 = rng.uniform(0, 1, m); X2 = rng.uniform(0, 1, m); S = np.abs(rng.normal(0, 1, m) * sd_eps(X1)); w = 4 * X2**3
                else:
                    X = rng.uniform(0, 1, m); S = np.abs(rng.normal(0, 1, m) * sd_eps(X)); w = v(X)
                tau = wquantile(S, w, beta)[0]
                Z[r] = np.sqrt(m) * (beta - np.interp(tau, tg, FQ_tab))
            res[str(beta)] = {"q": q, "Gtilde": Gt, "betaK2": beta * K2, "sigma2_pred": sigma2, "kish": kish, "emp_var": float(Z.var(ddof=1)),
                              "se_emp_var": float(np.std(Z**2, ddof=1) / np.sqrt(reps)), "eff_gain": (beta * K2 - Gt)**2 / (K2 - 1)}
        out["shifts"][kind] = res
    save("E8_coupling.json", out); return out

# ----------------------------------------------------------------------------------------------
# E9.  A learned predictor: cubic least squares fitted on a training split, then calibration asymptotics for the deployed model
# ----------------------------------------------------------------------------------------------
def E9(ntr=500, m=1000, reps=5000, reps_boot=2000, nboot=400, delta=0.1, levels=(0.8, 0.85, 0.9, 0.95)):
    global rng; rng = np.random.default_rng(10)
    sh = Shift("A"); K2 = sh.K2; Phi = stats.norm.cdf
    mu_true = lambda x: 2 * x + 0.5 * np.sin(6 * x)
    # training split from P, cubic least squares
    Xtr = sh.sampP(ntr); Ytr = mu_true(Xtr) + rng.normal(0, 1, ntr) * sd_eps(Xtr)
    coef = np.polyfit(Xtr, Ytr, 3); muhat = lambda x: np.polyval(coef, x)
    a = lambda x: mu_true(x) - muhat(x)                                     # Y - muhat(X) = a(X) + eps
    FQ = lambda t: integrate.quad(lambda x: sh.densQ(x) * (Phi((t - a(x)) / sd_eps(x)) - Phi((-t - a(x)) / sd_eps(x))), 0, 1, limit=200)[0]
    G = lambda t: integrate.quad(lambda x: sh.densQ(x) * sh.v(x) * (Phi((t - a(x)) / sd_eps(x)) - Phi((-t - a(x)) / sd_eps(x))), 0, 1, limit=200)[0]
    tg = np.linspace(0, 12, 2001); FQ_tab = np.array([FQ(t) for t in tg]); G_tab = np.array([G(t) for t in tg])
    FQi = lambda t: np.interp(t, tg, FQ_tab, right=1.0)
    b = np.array(levels); q = np.interp(b, FQ_tab, tg); Gt = np.interp(q, tg, G_tab); sigma2 = (1 - b)**2 * Gt + b**2 * (K2 - Gt)
    out = {"ntr": ntr, "m": m, "coef": coef.tolist(), "levels": levels, "sigma2_pred": sigma2.tolist(), "rmse_fit": float(np.sqrt(integrate.quad(lambda x: 4 * x**3 * a(x)**2, 0, 1)[0]))}
    Z = np.empty((reps, len(b)))
    for r in range(reps):
        X = sh.sampP(m); Y = mu_true(X) + rng.normal(0, 1, m) * sd_eps(X); S = np.abs(Y - muhat(X)); v = sh.v(X)
        tau = wquantile(S, v, b); Z[r] = np.sqrt(m) * (b - FQi(tau))
    out["emp_var"] = Z.var(0, ddof=1).tolist(); out["emp_mean"] = Z.mean(0).tolist()
    fails = {"none": 0, "pointwise": 0, "joint_boot": 0}; z1 = stats.norm.ppf(1 - delta)
    for r in range(reps_boot):
        X = sh.sampP(m); Y = mu_true(X) + rng.normal(0, 1, m) * sd_eps(X); S = np.abs(Y - muhat(X)); v = sh.v(X); vbar = v.mean()
        tau = wquantile(S, v, b); ind = (S[:, None] <= tau[None, :]).astype(float)
        Fhat = (v[:, None] * ind).sum(0) / v.sum(); H = v[:, None] * (ind - Fhat[None, :]); sig = np.sqrt((H**2).mean(0)) / vbar
        xi = rng.normal(0, 1, (nboot, m)); Zb = (xi @ H) / (np.sqrt(m) * vbar); chat = np.quantile(Zb.max(1), 1 - delta)
        def fail(infl):
            t2 = wquantile(S, v, b + infl); return bool(np.any(FQi(t2) < b))     # +inf threshold where the level exceeds one
        fails["none"] += fail(0.0); fails["pointwise"] += fail(z1 * sig / np.sqrt(m)); fails["joint_boot"] += fail(chat / np.sqrt(m))
    out["joint_failure"] = {k: val / reps_boot for k, val in fails.items()}; out["reps_boot"] = reps_boot
    save("E9_learned.json", out); return out

# ----------------------------------------------------------------------------------------------
# E10. Main-text experiment: estimated weights in a bounded parametric model (shift D).
#      P_X = Beta(theta_s,1), Q_X = Beta(theta_t,1), theta_s = 3, theta_t = 4, both estimated by maximum likelihood,
#      theta_hat = -n / sum(log X), from a target sample of size N_t and from a source sample that is either
#      independent of the calibration data (size N_s = m) or the calibration covariates themselves.
#      Predictions from Theorem 6.4 with gamma = E_Q[log X (F_{S|X}(q) - beta)]  (gamma_t = -gamma_s = gamma):
#        independent source:  tau^2 = sigma^2 + gamma^2 (theta_s^2 lambda_s + theta_t^2 lambda_t)
#        calibration source:  tau^2 = sigma^2 + gamma^2 (theta_t^2 lambda_t - theta_s^2)
#      For each configuration: empirical variance of sqrt(m)(beta - F_Q(tau_check(beta))), and failure probability
#      of the set run at beta + c/sqrt(m) for c the (1-delta)-quantile of the fixed-weight multiplier bootstrap
#      (naive) and of the pairs bootstrap (R draws, rank-corrected order statistic as in Procedure 1).
# ----------------------------------------------------------------------------------------------
def E10(ms=(500, 2000), reps=2000, nboot=400, delta=0.1, beta=0.8, seed=11, out_name="E10_estweights_bounded.json"):
    global rng; rng = np.random.default_rng(seed)
    sh = Shift("D"); ths, tht = sh.theta_s, sh.theta_t          # (the tabulation does not use rng)
    q = sh.qQ(beta); Gt = float(np.interp(q, sh.tgrid, sh.G_tab)); sigma2 = (1 - beta)**2 * Gt + beta**2 * (sh.K2 - Gt)
    gam = integrate.quad(lambda x: sh.densQ(x) * np.log(x) * (2 * stats.norm.cdf(q / sh.sd(x)) - 1 - beta), 0, 1, limit=200)[0]
    z1 = stats.norm.ppf(1 - delta)
    configs = [("oracle", None, None), ("indep", 1.0, 1.0), ("reuse", 4.0, None), ("reuse", 1.0, None), ("reuse", 0.25, None), ("reuse", 1 / 16, None)]
    settings = {"reps": reps, "nboot": nboot, "delta": delta, "beta": beta, "seed": seed}
    out = {"ms": list(ms), **settings, "sigma2": sigma2, "gamma": gam,
           "theta_s": ths, "theta_t": tht, "K2": sh.K2, "B": sh.B, "results": []}
    # Rows of an existing file for sample sizes that are not rerun are kept, but only if that file was produced with
    # the same settings; checked here, before any simulation.  (The stream of each sample size is seed*100000 + m.)
    path = os.path.join(RES, out_name); keep = []; old_ms = []
    if os.path.exists(path):
        old = json.load(open(path)); keep = [r for r in old.get("results", []) if r["m"] not in ms]
        old_settings = {k: old.get(k) for k in settings}
        if keep and old_settings != settings:
            raise ValueError(f"{out_name} holds results for m = {sorted(set(r['m'] for r in keep))} produced with {old_settings}; "
                             f"this run uses {settings}. Rerun those sample sizes too, or choose another out_name.")
        if keep: old_ms = old.get("ms", [])
    mle = lambda x: -len(x) / np.log(x).sum()
    mle_rows = lambda x: -x.shape[1] / np.log(x).sum(1)
    for m in ms:
        rng = np.random.default_rng(seed * 100000 + m)                # one stream per sample size: the results for a
        for src, lam_t, lam_s in configs:                              # given m do not depend on which other m are run
            Nt = int(round(m / lam_t)) if lam_t else None; Ns = int(round(m / lam_s)) if lam_s else None
            if src == "oracle": tau2 = sigma2
            elif src == "indep": tau2 = sigma2 + gam**2 * (ths**2 * lam_s + tht**2 * lam_t)
            else: tau2 = sigma2 + gam**2 * (tht**2 * lam_t - ths**2)
            Z = np.empty(reps); fail = {"naive": np.zeros(reps, bool), "pairs": np.zeros(reps, bool)}; bv = {"naive": np.empty(reps), "pairs": np.empty(reps)}
            for r in range(reps):
                X = sh.sampP(m); S = np.abs(rng.normal(0, 1, m) * sh.sd(X))
                if src == "oracle":
                    vh = sh.v(X)
                else:
                    Xt = sh.sampQ(Nt); Xs = sh.sampP(Ns) if src == "indep" else X
                    th_t, th_s = mle(Xt), mle(Xs); vh = (th_t / th_s) * X ** (th_t - th_s)
                tau = wquantile(S, vh, beta)[0]; Fc = (vh * (S <= tau)).sum() / vh.sum(); Z[r] = np.sqrt(m) * (beta - sh.FQ(tau))
                ind = (S <= tau).astype(float); H = vh * (ind - Fc)
                xi = rng.normal(0, 1, (nboot, m)); Zn = (xi @ H) / (np.sqrt(m) * vh.mean())
                if src == "oracle":
                    Lam = Zn   # with known weights the pairs bootstrap is the nonparametric bootstrap; use the multiplier version
                else:
                    idx = rng.integers(0, m, (nboot, m)); Xb = X[idx]; Sb = S[idx]
                    th_tb = mle_rows(Xt[rng.integers(0, Nt, (nboot, Nt))])
                    th_sb = mle_rows(Xs[rng.integers(0, Ns, (nboot, Ns))]) if src == "indep" else mle_rows(Xb)
                    vb = (th_tb / th_sb)[:, None] * Xb ** (th_tb - th_sb)[:, None]
                    Lam = np.sqrt(m) * ((vb * (Sb <= tau)).sum(1) / vb.sum(1) - Fc)
                for kind, draws in (("naive", Zn), ("pairs", Lam)):
                    c = rank_quantile(draws, delta); bv[kind][r] = draws.var(ddof=1)
                    lev = beta + c / np.sqrt(m); fail[kind][r] = bool(sh.FQ(wquantile(S, vh, lev)[0]) < beta)   # +inf threshold if lev > 1
            rec = {"m": m, "source": src, "Nt": Nt, "Ns": Ns, "tau2_pred": float(tau2), "emp_var": float(Z.var(ddof=1)),
                   "emp_var_se": float(Z.var(ddof=1) * np.sqrt(2 / (reps - 1))), "emp_mean": float(Z.mean()),
                   "boot_var": {k: float(v.mean()) for k, v in bv.items()},
                   "failure": {k: float(f.mean()) for k, f in fail.items()},
                   "failure_se": {k: float(np.sqrt(f.mean() * (1 - f.mean()) / reps)) for k, f in fail.items()},
                   "limit_failure_naive": float(1 - stats.norm.cdf(z1 * np.sqrt(sigma2 / tau2)))}
            out["results"].append(rec); print(json.dumps(rec))
    if keep:                                                           # merge with the rows kept for other sample sizes
        out["results"] = sorted(keep + out["results"], key=lambda r: r["m"])   # stable: configuration order kept
        out["ms"] = sorted(set(old_ms) | set(ms))
    save(out_name, out); return out

# ----------------------------------------------------------------------------------------------
# E11.  Airfoil benchmark: learned predictor, logistic-regression ratio, frozen-weight vs refitted bootstrap
# ----------------------------------------------------------------------------------------------
def load_airfoil():
    """UCI Airfoil Self-Noise data (Brooks, Pope and Marcolini, 1989; CC BY 4.0), see ../data/README.md.
    Columns: frequency, angle of attack, chord length, free-stream velocity, suction-side displacement thickness,
    scaled sound pressure level.  Frequency and thickness are log-transformed, as in Tibshirani et al. (2019)."""
    D = np.loadtxt(os.path.join(HERE, "..", "data", "airfoil_self_noise.dat"))
    X = D[:, :5].copy(); X[:, 0] = np.log(X[:, 0]); X[:, 4] = np.log(X[:, 4])
    return X, D[:, 5]

def logit_fit(ns, nt, Xt, XX, eta0=None, tol=1e-10, maxit=100):
    """Maximum likelihood for the logistic 'target versus source' model on a finite set of covariate points:
    row j of Xt (intercept and covariates) was drawn ns[..., j] times in the source sample and nt[..., j] times in
    the target sample.  ns, nt: arrays of shape (N,) or (R, N); R fits are run together by Newton's method.
    XX holds the N outer products of the rows of Xt, flattened."""
    ns = np.atleast_2d(ns).astype(float); nt = np.atleast_2d(nt).astype(float); n = ns + nt
    R, d = ns.shape[0], Xt.shape[1]
    eta = np.zeros((R, d)) if eta0 is None else np.broadcast_to(eta0, (R, d)).copy()
    for _ in range(maxit):
        p = 1.0 / (1.0 + np.exp(-(eta @ Xt.T)))
        g = (nt - n * p) @ Xt
        H = ((n * p * (1 - p)) @ XX).reshape(R, d, d)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        eta += step
        if np.abs(step).max() < tol: return eta
    raise RuntimeError("logit_fit: Newton's method did not converge")

def E11(ms=(500, 1000), nsplits=20, reps=250, nboot=400, delta=0.1, beta=0.9, tilt=0.5, ntrain=500, seed=12,
        out_name="E11_airfoil.json"):
    """Finite-population design on real data.  For each of `nsplits` random splits, the data are divided into a
    training set (least-squares predictor) and a population of N points.  P is the uniform law on the population and
    Q its exponential tilt, v(x) proportional to exp(tilt * (log thickness - log frequency)): the tilt of Tibshirani
    et al. (2019) at strength `tilt`.  Calibration and covariate samples are drawn with replacement, so they are
    i.i.d. from P and Q, every sample is a vector of counts on the population, and the coverage F_Q(tau) of a
    threshold is computed exactly.  The ratio is estimated by logistic regression of target versus source covariates;
    the source sample is either independent of the calibration sample or the calibration covariates themselves.
    `reps` calibration samples are drawn per split and every configuration is run on each of them; failure
    frequencies are pooled over the splits, and variances are reported relative to the variance with the true ratio,
    asymptotic (sigma^2 of the split) or empirical on the same samples."""
    X, y = load_airfoil(); n_all = len(y); N = n_all - ntrain; unif = np.full(N, 1.0 / N)
    z1 = stats.norm.ppf(1 - delta)
    configs = [("oracle", None, None), ("indep", 1.0, 1.0), ("reuse", 4.0, None), ("reuse", 1.0, None), ("reuse", 0.25, None)]
    kinds = ("frozen", "refit")
    out = {"ms": list(ms), "nsplits": nsplits, "reps": reps, "nboot": nboot, "delta": delta, "beta": beta, "tilt": tilt,
           "ntrain": ntrain, "seed": seed, "N": N, "splits": [], "results": []}
    pops = []
    for sp in range(nsplits):                                              # the populations, one per split
        perm = np.random.default_rng([seed, sp]).permutation(n_all); tr, pool = perm[:ntrain], perm[ntrain:]
        Atr = np.c_[np.ones(ntrain), X[tr]]; coef = np.linalg.lstsq(Atr, y[tr], rcond=None)[0]
        r2 = 1 - ((y[tr] - Atr @ coef)**2).sum() / ((y[tr] - y[tr].mean())**2).sum()
        Xp = X[pool]; S = np.abs(y[pool] - np.c_[np.ones(N), Xp] @ coef)
        o = np.argsort(S); S = S[o]; Xp = Xp[o]                           # population sorted by score
        assert len(np.unique(S)) == N
        Zc = (Xp - Xp.mean(0)) / Xp.std(0); Xt = np.c_[np.ones(N), Zc]; d = Xt.shape[1]
        XX = (Xt[:, :, None] * Xt[:, None, :]).reshape(N, d * d)
        w = np.exp(tilt * (Xp[:, 4] - Xp[:, 0])); v = w / w.mean(); qprob = v / N; K2 = float((v**2).mean())
        FQ_cum = np.cumsum(qprob); FQ_cum[-1] = 1.0
        kq = int(np.searchsorted(FQ_cum, beta)); Fq = float(FQ_cum[kq])   # population quantities at the level beta
        ind_q = (np.arange(N) <= kq).astype(float); h = v * (ind_q - Fq)
        sigma2 = float((h**2).mean()); Gt = float((v**2 * ind_q).mean())
        gam = (v[:, None] * (Zc - (v[:, None] * Zc).mean(0)) * (ind_q - Fq)[:, None]).mean(0)   # coupling gradient
        def tau2_pred(src, lam_t, lam_s, v=v, Xt=Xt, h=h, gam=gam, sigma2=sigma2):
            """asymptotic variance of the coverage error for the logistic estimator (Corollary 'coverage with
            estimated weights'), from the influence function of the two-sample logistic regression"""
            if src == "oracle": return sigma2
            Nt_m, Ns_m = 1.0 / lam_t, (1.0 / lam_s if src == "indep" else 1.0)   # sample sizes relative to m
            p = Nt_m * v / (Ns_m + Nt_m * v)
            A = Ns_m * (Xt.T * (p * (1 - p))) @ Xt / N + Nt_m * (Xt.T * (v * p * (1 - p))) @ Xt / N
            Ai = np.linalg.inv(A)
            gs = p[:, None] * Xt; gs = gs - gs.mean(0); gt = (1 - p)[:, None] * Xt; gt = gt - (v[:, None] * gt).mean(0)
            VarP = gs.T @ gs / N; VarQ = (gt.T * v) @ gt / N
            if src == "indep":
                return float(sigma2 + gam @ (Ai @ (Ns_m * VarP + Nt_m * VarQ) @ Ai.T)[1:, 1:] @ gam)
            psi = -(gs @ Ai.T)[:, 1:]; Saux = (Nt_m * Ai @ VarQ @ Ai.T)[1:, 1:]
            return float(sigma2 + 2 * gam @ (h[:, None] * psi).mean(0) + gam @ (psi.T @ psi / N + Saux) @ gam)
        kP = int(np.ceil(beta * N)) - 1                                   # unweighted population quantile
        pops.append(dict(S=S, Zc=Zc, Xt=Xt, XX=XX, v=v, qprob=qprob, FQ_cum=FQ_cum, sigma2=sigma2,
                         tau2={(c[0], c[1]): tau2_pred(*c) for c in configs}))
        out["splits"].append({"K2": K2, "max_v": float(v.max()), "max_atom_Q": float(qprob.max()), "r2_train": float(r2),
                              "q": float(S[kq]), "q_P": float(S[kP]), "FQ_at_q_P": float(FQ_cum[kP]), "sigma2": sigma2,
                              "sigma2_eff": float(sigma2 - (Fq * K2 - Gt)**2 / (K2 - 1)), "gamma": gam.tolist()})
    nc = len(configs); T = nsplits * reps
    for m in ms:
        # all configurations are run on the same calibration samples (and on nested target samples), so that the
        # comparisons between configurations and between the two bootstraps are paired
        Nts = sorted({int(round(m / c[1])) for c in configs if c[1]})    # target sample sizes, increasing
        Zs = np.empty((nc, T)); cover0 = np.empty((nc, T)); len0 = np.empty((nc, T))
        fail = {k: np.zeros((nc, T), bool) for k in kinds}; infl = {k: np.empty((nc, T)) for k in kinds}
        cover = {k: np.empty((nc, T)) for k in kinds}; length = {k: np.empty((nc, T)) for k in kinds}
        for sp, pop in enumerate(pops):
            rg = np.random.default_rng([seed, m, sp])                    # one stream per sample size and split
            S, Zc, Xt, XX, v, qprob, FQ_cum = (pop[k] for k in ("S", "Zc", "Xt", "XX", "v", "qprob", "FQ_cum"))
            def FQ(t):                                                    # exact coverage of the threshold t
                if not np.isfinite(t): return 1.0
                k = np.searchsorted(S, t, side="right"); return float(FQ_cum[k - 1]) if k > 0 else 0.0
            for r in range(sp * reps, (sp + 1) * reps):
                c = rg.multinomial(m, unif).astype(float)                 # calibration sample as counts
                ns_ind = rg.multinomial(m, unif)                          # independent source sample (N_s = m)
                nt_by = {}; acc = np.zeros(N, dtype=np.int64); prev = 0   # nested target samples
                for Nt in Nts:
                    acc = acc + rg.multinomial(Nt - prev, qprob); nt_by[Nt] = acc; prev = Nt
                for ci, (src, lam_t, lam_s) in enumerate(configs):
                    if src == "oracle":
                        vh = v
                    else:
                        Nt = int(round(m / lam_t)); nt = nt_by[Nt]; ns = c if src == "reuse" else ns_ind
                        eta = logit_fit(ns, nt, Xt, XX)[0]; vh = np.exp(Zc @ eta[1:])
                    wts = c * vh; tau = wquantile(S, wts, beta)[0]
                    ind = (S <= tau).astype(float); Fc = (wts * ind).sum() / wts.sum()
                    cover0[ci, r] = FQ(tau); len0[ci, r] = 2 * tau; Zs[ci, r] = np.sqrt(m) * (beta - cover0[ci, r])
                    # frozen weights: multiplier bootstrap of Procedure 1 (c_j standard normal multipliers sum to N(0, c_j))
                    xi = rg.normal(0, 1, (nboot, N)) * np.sqrt(c)
                    draws = {"frozen": (xi @ (vh * (ind - Fc))) / (np.sqrt(m) * wts.sum() / m)}
                    if src == "oracle":
                        draws["refit"] = draws["frozen"]                  # nothing to refit
                    else:                                                 # pairs bootstrap: resample and refit the ratio
                        cb = rg.multinomial(m, c / m, size=nboot).astype(float)
                        ntb = rg.multinomial(Nt, nt / Nt, size=nboot)
                        nsb = cb if src == "reuse" else rg.multinomial(m, ns / m, size=nboot)
                        vb = np.exp(logit_fit(nsb, ntb, Xt, XX, eta0=eta)[:, 1:] @ Zc.T) * cb
                        draws["refit"] = np.sqrt(m) * ((vb * ind).sum(1) / vb.sum(1) - Fc)
                    for kind in kinds:
                        cq = rank_quantile(draws[kind], delta); infl[kind][ci, r] = cq / np.sqrt(m)
                        t2 = wquantile(S, wts, beta + cq / np.sqrt(m))[0]   # +inf (whole space) if the level exceeds one
                        cover[kind][ci, r] = FQ(t2); fail[kind][ci, r] = cover[kind][ci, r] < beta
                        length[kind][ci, r] = 2 * t2 if np.isfinite(t2) else np.nan
        sig2 = np.array([pop["sigma2"] for pop in pops])
        var_split = Zs.reshape(nc, nsplits, reps).var(axis=2, ddof=1)     # empirical variances, per configuration and split
        for ci, (src, lam_t, lam_s) in enumerate(configs):
            Nt = int(round(m / lam_t)) if lam_t else None; Ns = int(round(m / lam_s)) if lam_s else None
            t2p = np.array([pop["tau2"][(src, lam_t)] for pop in pops])
            ratio = var_split[ci] / sig2                                  # over the asymptotic variance with the true ratio
            paired = var_split[ci] / var_split[0]                         # over the empirical variance with the true ratio, same samples
            dfail = fail["frozen"][ci].astype(float) - fail["refit"][ci]
            rec = {"m": m, "source": src, "Nt": Nt, "Ns": Ns,
                   "tau2_ratio_pred": float((t2p / sig2).mean()), "emp_var_ratio": float(ratio.mean()),
                   "emp_var_ratio_se": float(ratio.std(ddof=1) / np.sqrt(nsplits)),
                   "emp_var_ratio_paired": float(paired.mean()),
                   "emp_var_ratio_paired_se": float(paired.std(ddof=1) / np.sqrt(nsplits)),
                   "limit_failure_frozen": float((1 - stats.norm.cdf(z1 * np.sqrt(sig2 / t2p))).mean()),
                   "coverage_uninflated": float(cover0[ci].mean()), "failure_uninflated": float((cover0[ci] < beta).mean()),
                   "length_uninflated": float(len0[ci].mean()),
                   "failure": {k: float(f[ci].mean()) for k, f in fail.items()},
                   "failure_se": {k: float(np.sqrt(f[ci].mean() * (1 - f[ci].mean()) / T)) for k, f in fail.items()},
                   "failure_diff": float(dfail.mean()), "failure_diff_se": float(dfail.std(ddof=1) / np.sqrt(T)),
                   "inflation": {k: float(x[ci].mean()) for k, x in infl.items()},
                   "coverage": {k: float(x[ci].mean()) for k, x in cover.items()},
                   "length": {k: float(np.nanmean(x[ci])) for k, x in length.items()},
                   "length_diff": float(np.nanmean(length["frozen"][ci] - length["refit"][ci])),
                   "length_diff_se": float(np.nanstd(length["frozen"][ci] - length["refit"][ci], ddof=1) / np.sqrt(T)),
                   "whole_space": {k: int(np.isnan(x[ci]).sum()) for k, x in length.items()},
                   "tau2_ratio_pred_by_split": (t2p / sig2).tolist(), "emp_var_ratio_by_split": ratio.tolist()}
            out["results"].append(rec); print(json.dumps({k: v_ for k, v_ in rec.items() if not k.endswith("by_split")}), flush=True)
    save(out_name, out); return out

if __name__ == "__main__":
    t0 = time.time()
    import sys
    # The runs behind the stored files in results/ (one file each; E1, E4 and E5 also write a figure).  With no
    # argument all of them are executed.  Each seeds its own generator, so each can be run alone.
    runs = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6, "E7": E7,
            "E7m8000": lambda: E7(m=8000, reps=1500, cases=E7_CASES[:3], out_name="E7_estboot_m8000.json", seed=81),
            "E7b": E7b, "E8": E8, "E9": E9, "E10": E10, "E11": E11}
    names = sys.argv[1:] or list(runs)
    unknown = [nm for nm in names if nm not in runs]
    if unknown: sys.exit(f"unknown run(s) {unknown}; available: {' '.join(runs)}")
    for nm in names:
        t = time.time(); r = runs[nm](); print(nm, f"done in {time.time()-t:.0f}s"); print(json.dumps(r, indent=1)[:3000])
    print("total", f"{time.time()-t0:.0f}s")
