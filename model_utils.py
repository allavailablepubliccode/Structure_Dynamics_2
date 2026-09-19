"""Shared numerical utilities for the OU/CMI analyses."""
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm
from scipy.optimize import lsq_linear

JAC_EPS = 2e-4
JAC_REFRESH = 5
THETA_MIN = 0.50
THETA_MAX = 1.50
B_STEP = 0.012
PER_NODE_STEP = 0.004
LAG = 0.5


def spectral_radius(W):
    return float(np.max(np.abs(np.linalg.eigvals(W))))


def max_real_eig(A):
    return float(np.max(np.real(np.linalg.eigvals(A))))


def stable(A, margin=-1e-5):
    return max_real_eig(A) < margin


def drift(W, theta, gain, coupling):
    return coupling * np.diag(gain) @ W - np.diag(theta)


def gaussian_cmi_vector(W, theta, gain, noise_sd, coupling, lags=(LAG,)):
    """Analytic pairwise CMI vector for the stationary multivariate OU process.

    Ordering is: for each lag, receiver i=0..n-1, sender j=0..n-1, j != i.
    CMI is I(X_j(t); X_i(t+tau) | X_i(t)).
    """
    A = drift(W, theta, gain, coupling)
    if not stable(A):
        return None

    Sigma = np.diag(noise_sd)
    S = solve_continuous_lyapunov(A, -(Sigma @ Sigma.T))
    S = 0.5 * (S + S.T)
    variances = np.diag(S)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        return None

    n = W.shape[0]
    out = []
    for tau in lags:
        L = expm(A * float(tau)) @ S
        for i in range(n):
            var_y = variances[i]
            var_z = variances[i]
            cov_yz = L[i, i]
            for j in range(n):
                if j == i:
                    continue
                var_x = variances[j]
                cov_xy = S[j, i]
                cov_xz = L[i, j]

                rxy = cov_xy / np.sqrt(max(var_x * var_y, 1e-18))
                ryz = cov_yz / np.sqrt(max(var_y * var_z, 1e-18))
                rxz = cov_xz / np.sqrt(max(var_x * var_z, 1e-18))
                rxy = np.clip(rxy, -0.999999, 0.999999)
                ryz = np.clip(ryz, -0.999999, 0.999999)
                rxz = np.clip(rxz, -0.999999, 0.999999)

                den = np.sqrt(max((1-rxy*rxy)*(1-ryz*ryz), 1e-18))
                partial_r = np.clip((rxz-rxy*ryz)/den, -0.999999, 0.999999)
                out.append(-0.5*np.log(max(1-partial_r*partial_r, 1e-18)))

    out = np.asarray(out, dtype=float)
    return out if np.all(np.isfinite(out)) else None


def normalized_error(C, target):
    if C is None:
        return np.inf
    return float(np.linalg.norm(C-target) / max(np.linalg.norm(target), 1e-18))


def theta_jacobian(W, theta, gain, noise_sd, coupling, lags, eps=JAC_EPS):
    base = gaussian_cmi_vector(W, theta, gain, noise_sd, coupling, lags)
    if base is None:
        return None
    J = np.empty((base.size, theta.size), dtype=float)
    for k in range(theta.size):
        plus = theta.copy(); plus[k] += eps
        Cp = gaussian_cmi_vector(W, plus, gain, noise_sd, coupling, lags)
        if Cp is not None:
            J[:, k] = (Cp-base)/eps
        else:
            minus = theta.copy(); minus[k] -= eps
            Cm = gaussian_cmi_vector(W, minus, gain, noise_sd, coupling, lags)
            if Cm is None:
                return None
            J[:, k] = (base-Cm)/eps
    return J


def safe_backtrack(W, theta, dtheta, gain, noise_sd, coupling, lags):
    scale = 1.0
    trial = theta + dtheta
    for i in range(theta.size):
        if trial[i] < THETA_MIN and dtheta[i] < 0:
            scale = min(scale, (THETA_MIN-theta[i])/dtheta[i])
        if trial[i] > THETA_MAX and dtheta[i] > 0:
            scale = min(scale, (THETA_MAX-theta[i])/dtheta[i])
    scale = max(0.0, min(1.0, scale))
    while scale > 1e-5:
        C = gaussian_cmi_vector(W, theta + scale*dtheta, gain, noise_sd, coupling, lags)
        if C is not None:
            return scale*dtheta, C
        scale *= 0.5
    return np.zeros_like(dtheta), gaussian_cmi_vector(W, theta, gain, noise_sd, coupling, lags)


def adapt_one_step(W, theta, target, gain, noise_sd, coupling, lags,
                   B_step=B_STEP, per_node=PER_NODE_STEP, J=None):
    C_before = gaussian_cmi_vector(W, theta, gain, noise_sd, coupling, lags)
    if C_before is None:
        emergency = np.ones_like(theta)
        emergency *= B_step / np.linalg.norm(emergency)
        dtheta, C_after = safe_backtrack(W, theta, emergency, gain, noise_sd, coupling, lags)
        return theta+dtheta, C_after, None

    if J is None:
        J = theta_jacobian(W, theta, gain, noise_sd, coupling, lags)
    if J is None:
        return theta.copy(), C_before, None

    residual = C_before-target
    try:
        sol = lsq_linear(J, -residual, bounds=(-per_node, per_node), lsmr_tol="auto", max_iter=200)
        dtheta = sol.x
    except Exception:
        dtheta = np.zeros_like(theta)
    norm = np.linalg.norm(dtheta)
    if norm > B_step and norm > 0:
        dtheta *= B_step/norm
    dtheta, C_after = safe_backtrack(W, theta, dtheta, gain, noise_sd, coupling, lags)
    return theta+dtheta, C_after, J


def load_timescale_model(path):
    z = np.load(Path(path), allow_pickle=True)
    return {
        "W0": z["W0"].astype(float),
        "DeltaW": z["DeltaW"].astype(float),
        "labels": z["labels"].astype(str),
        "gain": z["gain"].astype(float),
        "noise_sd": z["noise_sd"].astype(float),
        "coupling": float(z["global_coupling"]),
        "target": z["C_target"].astype(float),
        "per_node": float(z["original_per_node"]),
        "lags": tuple(np.asarray(z["lags"], dtype=float).tolist()),
    }


def matching_index_binary(W):
    A = W > 0
    np.fill_diagonal(A, False)
    n = W.shape[0]
    M = np.zeros_like(W, dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            ni = A[i].copy(); nj = A[j].copy()
            ni[j] = False; nj[i] = False
            union = np.sum(ni | nj)
            M[i,j] = M[j,i] = np.sum(ni & nj)/union if union else 0.0
    return M
