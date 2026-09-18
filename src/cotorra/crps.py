#!/usr/bin/env python3

"""
CRPS (continuous ranked probability score) of a Beta-mixture CDF against a
point observation, via nested Gauss-Legendre / tanh-sinh quadrature --
used by loss.py's crps_loss (basis_blended_tokens.crps_loss: true)
"""

import math

import numpy as np
import torch as t

# outer (CRPS) quadrature: plain Gauss-Legendre, split exactly at the true
# rank r into [0,r]/[r,1] -- F(x) and (F(x)-1) are smooth, bounded CDFs, so
# GL converges fast here regardless of how extreme the mixture's Beta
# shape params are (validated in __main__ below).
DEFAULT_OUTER_POINTS = 8

# inner (CDF) quadrature: tanh-sinh (double-exponential), needed because a
# Beta(a,b) density with a<1 or b<1 has an *integrable but unbounded*
# singularity at 0 or 1 that ordinary Gauss-Legendre only resolves at an
# algebraic O(1/M) rate (M=256 nodes still ~0.2% off in __main__'s
# regression case) -- tanh-sinh's doubly-exponential node clustering near
# the endpoints handles this in a couple dozen nodes instead. h/s_max are
# NOT config-exposed (unlike DEFAULT_OUTER_POINTS): getting these two
# numerically right took three separate bugs to shake out (see
# tanh_sinh_rule's and beta_pdf's docstrings), so they're pinned here
# rather than left for a config file to silently mistune.
DEFAULT_INNER_H = 0.2
DEFAULT_INNER_S_MAX = 4.5

# CRPS-specific floor on the Beta shape params, *tighter* than
# BasisBlendedCausalLM._mixture_weights' own clamp(1e-3, 1e4) -- that
# floor is fine for computing mixture weights (a plain ratio), but even
# tanh-sinh's accuracy degrades as a shape param approaches the
# integrability boundary (a-1 -> -1): a=0.05 converges to machine
# precision with plain quadrature settings above, a=0.001 (the model's
# own floor) does not, even at ~1000 inner nodes (see __main__). Below
# 0.05, the CRPS term is left as a deliberately biased-but-finite,
# still-directionally-useful approximation rather than chasing exactness
# into a regime real trained models rarely reach anyway (the closest
# regression case elsewhere in this codebase, "alpha/beta combinations
# seen in real trained models," bottoms out at 0.33).
SHAPE_PARAM_CLAMP = (0.05, 1e4)


def gl_rule(m: int, dtype=t.float64) -> tuple[t.Tensor, t.Tensor]:
    """canonical Gauss-Legendre nodes/weights on [-1,1]."""
    nodes, weights = np.polynomial.legendre.leggauss(m)
    return t.tensor(nodes, dtype=dtype), t.tensor(weights, dtype=dtype)


def beta_component_interval_log_mass(a, b, lo, hi, nodes, weights, endpoint_eps=1e-6):
    """
    log of each Beta component's probability mass on [lo, hi]:

        log integral_lo^hi Beta_pdf(u; a_i, b_i) du      -> shape (..., k)

    Shared by two callers that need the same quantity, so the observation
    model is identical on both sides of the network:

    * Loss.basis_blended_token_loss's interval_nll_loss branch, which sums
      it against the model's *predicted* weights to score an observation.
    * BasisBlendedCausalLM._mixture_weights under
      interval_mixture_weights, which normalizes it into the *input*
      embedding blend.

    torch has no differentiable regularized incomplete beta, so this is a
    Gauss-Legendre rule mapped onto [lo, hi] -- a single definite integral
    over one interval, unlike crps' nested construction. Entirely in log
    space (logsumexp over nodes) because a sharply peaked Beta underflows
    to exactly 0 in linear space.

    `a`/`b` are (..., k); `lo`/`hi` are (...); returns (..., k).
    """
    half = ((hi - lo) / 2).unsqueeze(-1)  # (...,1)
    mid = ((hi + lo) / 2).unsqueeze(-1)
    u = (mid + half * nodes).clamp(endpoint_eps, 1.0 - endpoint_eps)  # (...,n_quad)
    uu = u.unsqueeze(-1)  # (...,n_quad,1)
    a_ = a.unsqueeze(-2)  # (...,1,k)
    b_ = b.unsqueeze(-2)
    log_pdf = (
        (a_ - 1.0) * t.log(uu)
        + (b_ - 1.0) * t.log1p(-uu)
        - (t.lgamma(a_) + t.lgamma(b_) - t.lgamma(a_ + b_))
    )  # (...,n_quad,k)
    log_gl = weights.log().unsqueeze(-1)  # (n_quad,1)
    return t.log(half.clamp_min(1e-30)) + t.logsumexp(log_gl + log_pdf, dim=-2)


def tanh_sinh_rule(
    h: float = DEFAULT_INNER_H, s_max: float = DEFAULT_INNER_S_MAX, dtype=t.float64
) -> tuple[t.Tensor, t.Tensor]:
    """
    Double-exponential (tanh-sinh) quadrature rule on canonical (-1,1).
    Returns (q, weights) where q = 1 + phi(s), phi(s) = tanh(pi/2 sinh(s))
    -- q itself, NOT phi, because computing phi first and then adding 1
    (to build x = (X/2)*(phi+1)) suffers catastrophic cancellation once
    phi is within float64 epsilon of -1, which is exactly the regime that
    matters here (resolving a Beta density's singularity as its shape
    param -> 0). The stable identity 1+tanh(y) = 2/(1+exp(-2y)) avoids the
    subtraction entirely; the analogous 1-tanh(y) = 2/(1+exp(2y)) gives
    dphi/ds = (pi/2 cosh(s)) * sech(y)^2 = (pi/2 cosh(s)) * (1-tanh(y))
    * (1+tanh(y)) without ever forming tanh(y) itself.

    Accuracy comes from a small fixed *step* h over a wide enough *range*
    s_max, not from raising a node count at fixed range: an earlier
    version of this function instead shrank h as a target node count grew
    (h = 4/m) while implicitly holding s_max roughly constant -- that
    plateaued at ~2-6% relative error on a pure x^(a-1) test regardless of
    how many nodes were requested, since more nodes just added redundant
    resolution near s=0 without ever reaching further toward the singular
    endpoint. Fixed h/s_max reaches ~1e-6 relative error on that same test
    with ~20 nodes. See __main__ for the full validation.
    """
    m_half = int(s_max / h)
    j = t.arange(-m_half, m_half + 1, dtype=dtype)
    s = j * h
    half_pi = t.tensor(np.pi / 2, dtype=dtype)
    y = half_pi * t.sinh(s)
    q = 2 / (1 + t.exp(-2 * y))  # = 1+tanh(y), stable as y -> -inf
    p = 2 / (1 + t.exp(2 * y))  # = 1-tanh(y), stable as y -> +inf
    dphi = (half_pi * t.cosh(s)) * p * q
    weights = h * dphi
    finite = t.isfinite(q) & t.isfinite(weights) & (weights > 0) & (q > 0) & (q < 2)
    return q[finite], weights[finite]


def beta_pdf(x: t.Tensor, a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """
    Beta(a,b) density at x, matching BasisBlendedCausalLM._mixture_weights'
    log-pdf formula exactly (same lgamma-based normalization), computed
    with clamp/sanitization bounds chosen to be safe in *any* dtype x
    arrives in, not just float64:

    - the lower clamp on x must not underflow to exactly 0 in x's own
      dtype -- a fixed constant like 1e-300 is fine in float64 but is a
      silent no-op in float32 (whose smallest positive value is ~1.4e-45,
      so anything that would clamp to 1e-300 just stays 0), producing
      log(0) = -inf upstream. t.finfo(x.dtype).tiny is always safe.
    - the nan_to_num posinf bound must stay safe to *exponentiate* in
      x's own dtype: exp(1e4) (a bound that's merely "big" in log-space)
      overflows float32 (max ~3.4e38, exp(88) already saturates it). A
      spurious +inf density at one quadrature node, multiplied by that
      same node's near-zero weight, is 0*inf=nan -- which then poisons
      the entire weighted sum via ordinary addition. exp(50) ~ 5.2e21 is
      still an astronomically dominant (never realistic) density but
      stays finite in float32.

    Both of these were real, reproduced failure modes (see __main__): the
    float64-only version of this function returned finite-but-silently-
    wrong CRPS values in float32 whenever a quadrature node's density
    happened to saturate the old, looser bounds.
    """
    tiny = t.finfo(x.dtype).tiny
    x = x.clamp(tiny, 1 - t.finfo(x.dtype).eps)
    log_pdf = (
        (a - 1) * x.log()
        + (b - 1) * (1 - x).log()
        - (t.lgamma(a) + t.lgamma(b) - t.lgamma(a + b))
    )
    log_pdf = t.nan_to_num(log_pdf, nan=-50.0, posinf=50.0, neginf=-50.0)
    return log_pdf.exp()


def mixture_crps(
    r: t.Tensor,
    w: t.Tensor,
    a: t.Tensor,
    b: t.Tensor,
    gl_out: tuple[t.Tensor, t.Tensor],
    de_in: tuple[t.Tensor, t.Tensor],
) -> t.Tensor:
    """
    CRPS(F, r) = integral_0^1 (F(x) - 1[x>=r])^2 dx for a Beta mixture CDF
    F(x) = sum_i w_i * I_x(a_i, b_i), against a point observation r --
    split exactly at r into integral_0^r F(x)^2 dx + integral_r^1
    (F(x)-1)^2 dx (r doesn't require grad, so this per-sample split is
    just an affine rescaling of the canonical outer rule, no autograd
    complication). F(x) itself has no closed form in torch (no incomplete
    beta function), so it's evaluated via a second, nested quadrature
    (tanh-sinh, robust to a<1/b<1 endpoint singularities -- see
    tanh_sinh_rule) at each outer node.

    r: (N,) true ranks, already clamped away from {0,1} by the caller
       (matching BasisBlendedCausalLM._mixture_weights' _RANK_EPS
       convention -- both outer pieces must be non-degenerate).
    w: (N,k) predicted mixture weights (sum to 1 over k) -- the model's
       own renormalized next-token distribution over the k basis tokens,
       same w used by the existing NLL term.
    a, b: (N,k) Beta shape params -- SHAPE_PARAM_CLAMP applied by the
       caller (tighter than _mixture_weights' own clamp; see that
       constant's docstring for why).
    gl_out: (nodes, weights) from gl_rule -- outer quadrature.
    de_in: (q, weights) from tanh_sinh_rule -- inner quadrature.

    Returns (N,) CRPS per position (always >= 0).
    """
    N, k = w.shape
    nodes_out, weights_out = gl_out
    q_in, weights_in = de_in

    r_ = r.unsqueeze(-1)  # (N,1)
    x0 = (r_ / 2) * (nodes_out.unsqueeze(0) + 1)  # (N,M_out) -- in [0,r]
    w0 = weights_out.unsqueeze(0) * (r_ / 2)
    x1 = r_ + ((1 - r_) / 2) * (nodes_out.unsqueeze(0) + 1)  # (N,M_out) -- in [r,1]
    w1 = weights_out.unsqueeze(0) * ((1 - r_) / 2)

    x_outer = t.stack([x0, x1], dim=1)  # (N,2,M_out) -- inner-quad upper bound
    w_outer = t.stack([w0, w1], dim=1)  # (N,2,M_out)

    x_outer_ = x_outer.unsqueeze(-1)  # (N,2,M_out,1)
    t_inner = (x_outer_ / 2) * q_in.view(1, 1, 1, -1)  # (N,2,M_out,M_in)
    w_inner = weights_in.view(1, 1, 1, -1) * (x_outer_ / 2)

    t_inner_ = t_inner.unsqueeze(-1)  # (N,2,M_out,M_in,1)
    a_ = a.view(N, 1, 1, 1, k)
    b_ = b.view(N, 1, 1, 1, k)
    pdf = beta_pdf(t_inner_, a_, b_)  # (N,2,M_out,M_in,k)

    w_inner_ = w_inner.unsqueeze(-1)
    cdf_per_component = (pdf * w_inner_).sum(dim=3)  # (N,2,M_out,k)

    w_ = w.view(N, 1, 1, k)
    F = (cdf_per_component * w_).sum(dim=-1)  # (N,2,M_out)
    # quadrature approximation error can push F fractionally outside
    # [0,1] (a real CDF never is); clamp for consistency with the CRPS
    # integral's own definition, which assumes a valid CDF.
    F = F.clamp(0.0, 1.0)

    sq_err = t.stack([F[:, 0, :] ** 2, (F[:, 1, :] - 1) ** 2], dim=1)  # (N,2,M_out)
    return (sq_err * w_outer).sum(dim=(1, 2))  # (N,)


if __name__ == "__main__":
    from scipy.integrate import quad
    from scipy.stats import beta as scipy_beta

    def _reference_crps(r, w, a, b):
        def F(x):
            return sum(wi * scipy_beta.cdf(x, ai, bi) for wi, ai, bi in zip(w, a, b))

        def integrand(x):
            return (F(x) - (1.0 if x >= r else 0.0)) ** 2

        val, _ = quad(integrand, 0, 1, points=[r], limit=200)
        return val

    k = 10
    idx = np.arange(k)
    a_init, b_init = idx + 1.0, k - idx  # order-statistic init
    # "alpha/beta combinations seen in real trained models" -- copied
    # directly from BasisBlendedCausalLM.__main__'s own regression test
    a_extreme = np.array([0.33, 0.80, 1.79, 3.34, 13.85, 18.70, 20.97, 19.56, 5.0, 2.0])
    b_extreme = np.array([24.72, 21.73, 16.28, 10.70, 1.45, 8.48, 0.79, 0.49, 3.0, 6.0])
    # near SHAPE_PARAM_CLAMP's own floor, both a<1 and b<1 present
    a_vext = np.array([0.05, 0.95, 2.0, 5.0, 0.1, 15.0, 3.0, 8.0, 1.0, 0.5])
    b_vext = np.array([30.0, 0.08, 10.0, 3.0, 20.0, 0.5, 6.0, 2.0, 1.0, 12.0])

    test_cases = (
        [(r, np.ones(k) / k, a_init, b_init) for r in (0.001, 0.13, 0.5, 0.9, 0.999)]
        + [
            (r, np.ones(k) / k, a_extreme, b_extreme)
            for r in (0.001, 0.1, 0.5, 0.9, 0.999)
        ]
        + [(r, np.ones(k) / k, a_vext, b_vext) for r in (0.001, 0.05, 0.5, 0.95, 0.999)]
    )

    for dtype, label in [
        (t.float64, "float64"),
        (t.float32, "float32 (training dtype)"),
    ]:
        gl_out = tuple(x.to(dtype) for x in gl_rule(DEFAULT_OUTER_POINTS))
        de_in = tuple(x.to(dtype) for x in tanh_sinh_rule())
        max_err, n_bad = 0.0, 0
        for r, w, a, b in test_cases:
            ref = _reference_crps(r, w, a, b)
            a_c, b_c = np.clip(a, *SHAPE_PARAM_CLAMP), np.clip(b, *SHAPE_PARAM_CLAMP)
            got = mixture_crps(
                t.tensor([r], dtype=dtype),
                t.tensor(np.array([w]), dtype=dtype),
                t.tensor(np.array([a_c]), dtype=dtype),
                t.tensor(np.array([b_c]), dtype=dtype),
                gl_out,
                de_in,
            ).item()
            if not np.isfinite(got):
                n_bad += 1
                continue
            max_err = max(max_err, abs(got - ref) / max(ref, 1e-8))
        assert n_bad == 0, f"{label}: {n_bad} non-finite CRPS value(s)"
        assert max_err < 1e-2, f"{label}: max relative error {max_err:.2e} too high"
        print(
            f"{label}: {len(test_cases)} cases finite, "
            f"max relative error {max_err:.2e}: OK"
        )

    # gradient check (float32, matching training dtype)
    gl_out = tuple(x.float() for x in gl_rule(DEFAULT_OUTER_POINTS))
    de_in = tuple(x.float() for x in tanh_sinh_rule())
    r_t = t.tensor([0.13, 0.8, 0.02, 0.98], dtype=t.float32)
    w_leaf = (t.randn(4, k) * 1.0).requires_grad_()
    a_leaf = (t.randn(4, k) * 3.0).requires_grad_()
    b_leaf = (t.randn(4, k) * 3.0).requires_grad_()
    w_t = t.softmax(w_leaf, dim=-1)
    a_t = a_leaf.exp().clamp(*SHAPE_PARAM_CLAMP)
    b_t = b_leaf.exp().clamp(*SHAPE_PARAM_CLAMP)
    crps = mixture_crps(r_t, w_t, a_t, b_t, gl_out, de_in)
    assert t.isfinite(crps).all()
    crps.sum().backward()
    for name, leaf in [("w", w_leaf), ("a", a_leaf), ("b", b_leaf)]:
        assert t.isfinite(leaf.grad).all(), f"{name}: non-finite gradient"
        assert leaf.grad.abs().sum() > 0, f"{name}: zero gradient"
    print("gradient reaches w/a/b through mixture_crps: OK")

    print("all crps.py self-tests passed")


# --------------------------------------------------------------------------
# Truncated-normal component family (component_family: truncnorm)
#
# WHY THIS EXISTS. Beta has a structural hole at the ends of the rank range:
# Beta(a,b) is proportional to x^(a-1) (1-x)^(b-1), so its density at x=1 is
# EXACTLY zero whenever b>1, and at x=0 whenever a>1 -- no choice of mixture
# weights can put mass there. Measured on trained checkpoints this is not
# hypothetical: 54-102 of 151 categories per model had no active component
# with b<1 (dead at r->1) and 64-103 had none with a<1 (dead at r->0), while
# 1.03% of held-out numeric observations sit at r>0.99 and 0.70% at r<0.01.
# The best log-density ANY weighting could reach at the worst-covered rank ran
# from -57 (k=10) to -249 (k=32), against a decile baseline that cannot fall
# below about -1.97 by construction. That is the mechanism behind the basis
# models' heavy NLL tail (p99 up to +7.27 nats) despite BETTER medians.
#
# A truncated normal on [0,1] is strictly positive everywhere on the interval
# for any finite (mu, sigma), and -- the part that matters -- mu may sit AT or
# BEYOND a boundary, giving a monotone "mass piles up at the top" shape. Beta
# can only approximate that with b<1, an unbounded endpoint spike, which is
# why SHAPE_PARAM_CLAMP floors shapes at 0.05.
#
# It is also cheaper. beta_component_interval_log_mass needs 32-node
# Gauss-Legendre because the incomplete beta has no elementary form; the
# interval mass below is closed form in erf, so the interval NLL -- the
# primary training objective -- stops being a quadrature approximation. For
# the same reason truncnorm_mixture_crps needs only the OUTER quadrature, not
# the nested inner one mixture_crps requires.
#
# NOT a complete fix on its own: a very sharp component still gives tiny
# density far from its mode, so the tail is bounded only by sigma's floor. The
# hard guarantee is a uniform floor component, which is separate.
# --------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
SIGMA_CLAMP = (1e-3, 10.0)
# mu is deliberately allowed OUTSIDE [0,1]: a component peaking at or
# beyond a boundary is exactly what gives this family the edge coverage
# Beta cannot express. These bounds only stop a divergent run from
# reaching inf/NaN; they never bind on a useful value.
MU_CLAMP = (-10.0, 10.0)


def _std_normal_interval_prob(za: t.Tensor, zb: t.Tensor) -> t.Tensor:
    """
    Phi(zb) - Phi(za) without catastrophic cancellation.

    The naive 0.5*(erf(zb/sqrt2) - erf(za/sqrt2)) loses all precision when both
    ends sit far out in the SAME tail -- exactly where a sharp component far
    from the observation lands, and exactly where the resulting log mass
    matters most. Using erfc on the right tail keeps the subtraction between
    two small numbers instead of two numbers near 1.
    """
    right = 0.5 * (t.erfc(za / _SQRT2) - t.erfc(zb / _SQRT2))
    # mirror image for the LEFT tail. Without this branch, mu>1 with a small
    # sigma puts both ends at large negative z, erf saturates at -1, and the
    # difference cancels to exactly 0 -- which silently kills precisely the
    # edge-peaked component this family exists to make possible (it showed up
    # as mixture CDF F(1) = 0.607 instead of 1).
    left_tail = 0.5 * (t.erfc(-zb / _SQRT2) - t.erfc(-za / _SQRT2))
    straddle = 0.5 * (t.erf(zb / _SQRT2) - t.erf(za / _SQRT2))
    return t.where(za > 0, right, t.where(zb < 0, left_tail, straddle))


def _trunc_norm_denom(mu, sigma, support):
    A, B = support
    return _std_normal_interval_prob((A - mu) / sigma, (B - mu) / sigma)


def trunc_normal_interval_log_mass(mu, sigma, lo, hi, support=(0.0, 1.0)):
    """
    log of each component's probability mass on [lo, hi] -- the closed-form
    drop-in for beta_component_interval_log_mass, no nodes/weights needed.
    mu/sigma (..., k); lo/hi (...,); returns (..., k).
    """
    lo_ = lo.unsqueeze(-1)
    hi_ = hi.unsqueeze(-1)
    num = _std_normal_interval_prob((lo_ - mu) / sigma, (hi_ - mu) / sigma)
    den = _trunc_norm_denom(mu, sigma, support)
    tiny = t.finfo(num.dtype).tiny
    return num.clamp_min(tiny).log() - den.clamp_min(tiny).log()


def trunc_normal_log_pdf(x, mu, sigma, support=(0.0, 1.0)):
    """log density of each component at x. x (...,) -> (..., k)."""
    x_ = x.unsqueeze(-1)
    z = (x_ - mu) / sigma
    log_phi = -0.5 * z * z - 0.5 * math.log(2 * math.pi) - sigma.log()
    den = _trunc_norm_denom(mu, sigma, support)
    return log_phi - den.clamp_min(t.finfo(z.dtype).tiny).log()


def trunc_normal_mixture_cdf(x, w, mu, sigma, support=(0.0, 1.0)):
    """mixture CDF F(x) = sum_i w_i * TruncNormCDF_i(x), closed form."""
    A, _ = support
    x_ = x.unsqueeze(-1)
    num = _std_normal_interval_prob((A - mu) / sigma, (x_ - mu) / sigma)
    den = _trunc_norm_denom(mu, sigma, support)
    tiny = t.finfo(num.dtype).tiny
    return (w * (num / den.clamp_min(tiny)).clamp(0.0, 1.0)).sum(-1)


def truncnorm_mixture_crps(r, w, mu, sigma, gl_out, support=(0.0, 1.0)):
    """
    CRPS(F, r) for a truncated-normal mixture. Same split at r as
    mixture_crps, but F is closed form so only the OUTER rule is needed.
    r (N,), w/mu/sigma (N,k) -> (N,).
    """
    nodes, weights = (x.to(r.dtype).to(r.device) for x in gl_out)
    r_ = r.unsqueeze(-1)
    x0 = (r_ / 2) * (nodes.unsqueeze(0) + 1)
    w0 = weights.unsqueeze(0) * (r_ / 2)
    x1 = r_ + ((1 - r_) / 2) * (nodes.unsqueeze(0) + 1)
    w1 = weights.unsqueeze(0) * ((1 - r_) / 2)
    mu_, sd_, w_ = mu.unsqueeze(1), sigma.unsqueeze(1), w.unsqueeze(1)
    F0 = trunc_normal_mixture_cdf(x0, w_, mu_, sd_, support)
    F1 = trunc_normal_mixture_cdf(x1, w_, mu_, sd_, support)
    return (F0**2 * w0).sum(-1) + ((F1 - 1) ** 2 * w1).sum(-1)
