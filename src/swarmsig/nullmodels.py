"""Dynamikfreie Nullmodelle zur Kalibrierung der Schätzer (7.7).

| ID     | Modell                                                         |
|--------|----------------------------------------------------------------|
| N∅     | homogenes Feld: alle Individuen mit identischer Geschwindigkeit |
| N0     | unabhängige Schwankungen (stationärer OU-Schnappschuss)         |
| N1     | Gauß-Feld mit Matérn-Kovarianz fester Längen (ℓ_s, ℓ_⊥)         |
| N2     | wie N1, aber ℓ_s = c_s L, ℓ_⊥ = c_⊥ L                           |
| N2-mix | ℓ_s fest, ℓ_⊥ ∝ L („D1-artig“) bzw. beide ∝ L („B1-artig“)       |
| N3     | N1 + starre Rotation + Dilatation bekannter Amplitude          |

Positionen werden mit wählbarer Dichte erzeugt (vogelartig dicht oder drohnenartig dünn).
Das Speed-Feld δs wirkt längs der mittleren Flugrichtung n̂, das Richtungsfeld besteht aus
zwei unabhängigen Komponenten senkrecht zu n̂.

Gauß-Felder: exakt per Cholesky bis ``cholesky_max_n`` Punkte, darüber per Zufalls-
Fourier-Merkmalen (Spektraldichte der Matérn-Kovarianz = mehrdimensionale t-Verteilung mit
2ν Freiheitsgraden, skaliert mit 1/ℓ). Matérn-Glattheit ν: **ANNAHME** (Config).
Zeitreihen der Nullmodelle (für P2a) folgen mit dem Radar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma, kv

from .dynamics.init import ellipsoid_positions
from .observables.size import L_g


def matern(r: np.ndarray, ell: float, nu: float, sigma: float = 1.0) -> np.ndarray:
    """Matérn-Kovarianz C(r) = σ² 2^{1−ν}/Γ(ν) (√(2ν) r/ℓ)^ν K_ν(√(2ν) r/ℓ), C(0) = σ²."""
    r = np.asarray(r, dtype=float)
    q = np.sqrt(2.0 * nu) * r / ell
    if nu == 0.5:
        c = np.exp(-q)
    elif nu == 1.5:
        c = (1.0 + q) * np.exp(-q)
    elif nu == 2.5:
        c = (1.0 + q + q**2 / 3.0) * np.exp(-q)
    else:
        with np.errstate(invalid="ignore"):
            c = np.where(q > 0, 2 ** (1 - nu) / gamma(nu) * q**nu * kv(nu, np.maximum(q, 1e-300)), 1.0)
    return sigma**2 * c


def hole_kernel(r: np.ndarray, ell: float) -> np.ndarray:
    """Kovarianz mit Integral null: C(r) = (1 − r²/(3ℓ²)) exp(−r²/(2ℓ²)), Nullstelle bei √3·ℓ.

    Positiv definit in 3D (Spektraldichte ∝ k² exp(−k²ℓ²/2)). Ergänzung zu 7.7: Weil
    ∫C d³r = 0, entsteht durch den Mittelwertabzug kein Versatz; die geschätzte Nullstelle
    hängt dann nicht von L ab. Bei Matérn-Feldern wandert sie dagegen logarithmisch mit L
    (Versatz ≈ ∫C d³r / V), siehe Test ``test_n1_matern_drifts_hole_does_not``.
    """
    r = np.asarray(r, dtype=float)
    return (1.0 - r**2 / (3.0 * ell**2)) * np.exp(-(r**2) / (2.0 * ell**2))


def gaussian_field(x: np.ndarray, ell: float, nu: float, rng: np.random.Generator, *,
                   n_fields: int = 1, cholesky_max_n: int = 4000, n_features: int = 4096,
                   kernel: str = "matern") -> np.ndarray:
    """``n_fields`` unabhängige skalare Gauß-Felder (Varianz 1) an den Punkten ``x``; Form (n, n_fields).

    ``kernel``: "matern" (7.7, Glattheit ν) oder "hole" (:func:`hole_kernel`).
    """
    if kernel not in ("matern", "hole"):
        raise ValueError(f"unbekannter Kern {kernel!r}")
    n = x.shape[0]
    if n <= cholesky_max_n:
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        K = matern(d, ell, nu) if kernel == "matern" else hole_kernel(d, ell)
        jitter = 1e-10
        while True:
            try:
                Lc = np.linalg.cholesky(K + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > 1e-3:
                    raise
        return Lc @ rng.standard_normal((n, n_fields))
    # Zufalls-Fourier-Merkmale. Matérn: k = t/ℓ, t ~ t-Verteilung (df = 2ν) in 3D.
    # Loch-Kern: |k| = χ₅/ℓ (Radialdichte ∝ k⁴ exp(−k²ℓ²/2)), Richtung gleichverteilt.
    out = np.empty((n, n_fields))
    for f in range(n_fields):
        z = rng.standard_normal((n_features, 3))
        if kernel == "matern":
            chi2 = rng.chisquare(2.0 * nu, size=n_features)
            k = z / np.sqrt(chi2 / (2.0 * nu))[:, None] / ell
        else:
            radius = np.sqrt(rng.chisquare(5.0, size=n_features)) / ell
            k = z / np.linalg.norm(z, axis=1, keepdims=True) * radius[:, None]
        phase = rng.uniform(0.0, 2 * np.pi, size=n_features)
        out[:, f] = np.sqrt(2.0 / n_features) * np.cos(x @ k.T + phase).sum(axis=1)
    return out


def _frame(n_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zwei Einheitsvektoren senkrecht zu n̂."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n_hat, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n_hat, e1)


@dataclass(frozen=True)
class NullSnapshot:
    model: str
    x: np.ndarray
    u: np.ndarray
    ell_s: float = float("nan")
    ell_perp: float = float("nan")


def positions(n: int, spacing: float, rng: np.random.Generator,
              aspect: tuple[float, float, float] = (1.0, 1.0, 0.5),
              heading: np.ndarray | None = None) -> np.ndarray:
    """Zufällige Positionen in einem Ellipsoid mit Dichte 1/spacing³ (vogel- oder drohnenartig)."""
    return ellipsoid_positions(n, np.zeros(3), spacing, aspect,
                               np.array([0.0, 1.0, 0.0]) if heading is None else heading, rng)


def n_empty(x: np.ndarray, U: np.ndarray) -> NullSnapshot:
    """N∅: homogenes Geschwindigkeitsfeld."""
    return NullSnapshot("N_empty", x, np.broadcast_to(np.asarray(U, float), x.shape).copy())


def n0(x: np.ndarray, U: np.ndarray, sigma_s: float, sigma_perp: float,
       rng: np.random.Generator) -> NullSnapshot:
    """N0: unabhängige Schwankungen (Schnappschuss eines stationären OU-Prozesses)."""
    U = np.asarray(U, float)
    n_hat = U / np.linalg.norm(U)
    e1, e2 = _frame(n_hat)
    z = rng.standard_normal((x.shape[0], 3))
    u = U + sigma_s * z[:, :1] * n_hat + sigma_perp * (z[:, 1:2] * e1 + z[:, 2:3] * e2)
    return NullSnapshot("N0", x, u)


def n1(x: np.ndarray, U: np.ndarray, ell_s: float, ell_perp: float, sigma_s: float,
       sigma_perp: float, rng: np.random.Generator, *, nu: float = 0.5,
       cholesky_max_n: int = 4000, n_features: int = 4096, kernel: str = "matern",
       model: str = "N1") -> NullSnapshot:
    """N1: Gauß-Feld mit Matérn-Kovarianz der Längen ℓ_s (Speed) und ℓ_⊥ (Richtung)."""
    U = np.asarray(U, float)
    n_hat = U / np.linalg.norm(U)
    e1, e2 = _frame(n_hat)
    kw = dict(cholesky_max_n=cholesky_max_n, n_features=n_features, kernel=kernel)
    fs = gaussian_field(x, ell_s, nu, rng, n_fields=1, **kw)
    fp = gaussian_field(x, ell_perp, nu, rng, n_fields=2, **kw)
    u = U + sigma_s * fs[:, :1] * n_hat + sigma_perp * (fp[:, :1] * e1 + fp[:, 1:2] * e2)
    return NullSnapshot(model, x, u, ell_s, ell_perp)


def n2(x: np.ndarray, U: np.ndarray, c_s: float, c_perp: float, sigma_s: float,
       sigma_perp: float, rng: np.random.Generator, **kw: float) -> NullSnapshot:
    """N2: skalenfreie Variante mit ℓ = c·L (L = L_g der Positionen)."""
    L = L_g(x)
    return n1(x, U, c_s * L, c_perp * L, sigma_s, sigma_perp, rng, model="N2", **kw)  # type: ignore[arg-type]


def n2_mix(x: np.ndarray, U: np.ndarray, ell_s_fixed: float, c_perp: float, sigma_s: float,
           sigma_perp: float, rng: np.random.Generator, **kw: float) -> NullSnapshot:
    """N2-mix „D1-artig“: ℓ_s fest, ℓ_⊥ ∝ L (die „B1-artige“ Variante ist :func:`n2`)."""
    L = L_g(x)
    return n1(x, U, ell_s_fixed, c_perp * L, sigma_s, sigma_perp, rng, model="N2-mix", **kw)  # type: ignore[arg-type]


def add_collective_modes(x: np.ndarray, u: np.ndarray, Omega: np.ndarray, D: float) -> np.ndarray:
    """u + Ω × (x − x̄) + D (x − x̄)  (für N3)."""
    r = x - x.mean(axis=0)
    return u + np.cross(np.asarray(Omega, float)[None, :], r) + D * r


def n3(base: NullSnapshot, Omega: np.ndarray, D: float) -> NullSnapshot:
    """N3: N1 + starre Rotation + Dilatation bekannter Amplitude."""
    return NullSnapshot("N3", base.x, add_collective_modes(base.x, base.u, Omega, D),
                        base.ell_s, base.ell_perp)
