"""
psf_estimation.py
-----------------
Estimation des paramètres d'une PSF de flou de mouvement linéaire
à partir de la seule image floue.

MÉTHODE PRINCIPALE — CEPSTRE (Cannon, 1976)
-------------------------------------------
Le cepstre réel d'une image floue est défini par :

        C(x,y) = F⁻¹{ log |F(y)| } (x,y)

Si  y = h * x + n  avec h une PSF de flou de mouvement rectiligne de
longueur L à l'angle θ, on montre que le cepstre présente DEUX PICS
NÉGATIFS symétriques, situés à ±L pixels de l'origine, alignés selon
la direction du mouvement. C'est la signature classique qui permet
d'estimer simultanément L et θ en cherchant le minimum du cepstre
hors du pic central (DC).

MÉTHODE SECONDAIRE — SPECTRE DE FOURIER (pédagogique)
-----------------------------------------------------
La TF d'une PSF rectiligne est un sinc cardinal dont les zéros forment
des BANDES SOMBRES perpendiculaires à la direction du mouvement,
espacées d'une distance Δ ≈ N/L dans un spectre de taille N. On
peut donc estimer θ (orientation des bandes) puis L (par détection
des creux du profil). Cette seconde méthode est fournie à des fins
de comparaison et de visualisation.

Ces deux méthodes proviennent directement de l'analyse fréquentielle
vue en cours (FFT, convolution dans le domaine de Fourier) et ne
reposent sur aucun entraînement de réseau de neurones.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Pré-traitement commun
# ---------------------------------------------------------------------------

def _to_gray_square(image: np.ndarray) -> np.ndarray:
    """
    Convertit l'image en niveaux de gris dans [0,1], la centre
    (moyenne 0) et la pad dans un carré de taille N = max(H, W) (paire)
    avec un fenêtrage Hann 2D pour supprimer les discontinuités de bord.
    """
    if image.ndim == 3:
        g = (0.299 * image[..., 2] + 0.587 * image[..., 1]
             + 0.114 * image[..., 0])
    else:
        g = image.astype(np.float64)
    g = g.astype(np.float64)
    if g.max() > 1.5:
        g = g / 255.0

    H, W = g.shape
    N = max(H, W)
    if N % 2:
        N += 1
    padded = np.zeros((N, N), dtype=np.float64)
    y0 = (N - H) // 2
    x0 = (N - W) // 2
    padded[y0:y0 + H, x0:x0 + W] = g - g.mean()

    wy = np.hanning(N)[:, None]
    wx = np.hanning(N)[None, :]
    padded *= wy * wx
    return padded


def log_spectrum(image: np.ndarray) -> np.ndarray:
    """log(1 + |F|) centré (fftshift), pour visualisation."""
    padded = _to_gray_square(image)
    F = np.fft.fftshift(np.fft.fft2(padded))
    return np.log1p(np.abs(F))


# ---------------------------------------------------------------------------
# Méthode 1 — CEPSTRE (principale, robuste)
# ---------------------------------------------------------------------------

def compute_cepstrum(image: np.ndarray) -> np.ndarray:
    """Cepstre réel, centré par fftshift."""
    padded = _to_gray_square(image)
    F = np.fft.fft2(padded)
    logF = np.log(np.abs(F) + 1e-6)
    C = np.real(np.fft.ifft2(logF))
    return np.fft.fftshift(C)


def estimate_psf_cepstrum(image: np.ndarray,
                          r_min: float = 4.0,
                          r_max_frac: float = 0.45) -> dict:
    """
    Estime (L, θ) à partir du cepstre en cherchant son minimum
    en dehors d'un petit disque central (DC).
    """
    C = compute_cepstrum(image)
    N = C.shape[0]
    cy = cx = N // 2

    yy, xx = np.ogrid[:N, :N]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (r > r_min) & (r < N * r_max_frac)
    C_masked = np.where(mask, C, 0.0)

    idx = int(np.argmin(C_masked))
    py, px = divmod(idx, N)
    dx = px - cx
    dy = py - cy

    length = int(round(np.sqrt(dx * dx + dy * dy)))
    if length < 3:
        length = 3
    # θ = angle du vecteur (dx, -dy) ramené dans [0, 180)
    angle = float(np.degrees(np.arctan2(-dy, dx)) % 180.0)

    return {
        "length": length,
        "angle_deg": angle,
        "cepstrum": C,
        "peak_xy": (px, py),
        "method": "cepstrum",
    }


# ---------------------------------------------------------------------------
# Méthode 2 — ANALYSE SPECTRALE (pédagogique)
# ---------------------------------------------------------------------------

def estimate_angle_spectrum(image: np.ndarray,
                            n_angles: int = 360
                            ) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Intègre |F| le long de droites passant par le centre (projection
    de type Radon) après normalisation radiale, puis renvoie l'angle
    maximum.
    """
    spec = log_spectrum(image)
    N = spec.shape[0]
    cy = cx = N // 2

    yy, xx = np.ogrid[:N, :N]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_int = r.astype(int)
    r_max = min(cy, cx) - 2
    spec_norm = np.zeros_like(spec)
    for ri in range(r_max + 1):
        m = (r_int == ri)
        if m.any():
            mu = spec[m].mean()
            if mu > 0:
                spec_norm[m] = spec[m] / mu

    mask = (r > N * 0.08) & (r < N * 0.4)
    spec_use = spec_norm * mask

    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    ts = np.arange(-r_max, r_max + 1)
    scores = np.zeros(n_angles)
    for i, a in enumerate(angles):
        theta = np.deg2rad(a)
        xs = np.clip(np.round(cx + ts * np.cos(theta)).astype(int), 0, N - 1)
        ys = np.clip(np.round(cy + ts * np.sin(theta)).astype(int), 0, N - 1)
        scores[i] = spec_use[ys, xs].sum()

    best = int(np.argmax(scores))
    return float(angles[best]), angles, scores


def estimate_length_from_profile(image: np.ndarray, angle_deg: float
                                 ) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Profil de |F| perpendiculairement au mouvement → espacement des
    creux → L = N/Δ.
    """
    spec = log_spectrum(image)
    N = spec.shape[0]
    cy = cx = N // 2

    theta_perp = np.deg2rad(angle_deg + 90.0)
    R = min(cy, cx) - 2
    ts = np.arange(-R, R + 1)
    xs = np.clip(np.round(cx + ts * np.cos(theta_perp)).astype(int), 0, N - 1)
    ys = np.clip(np.round(cy + ts * np.sin(theta_perp)).astype(int), 0, N - 1)
    profile = spec[ys, xs].astype(np.float64)

    t = np.arange(len(profile))
    trend = np.polyval(np.polyfit(t, profile, 3), t)
    detr = profile - trend

    center = len(profile) // 2
    guard = max(5, int(0.02 * len(profile)))
    detr_g = detr.copy()
    detr_g[center - guard: center + guard + 1] = 0.0

    inv = -detr_g
    peaks, _ = find_peaks(inv, distance=3, prominence=inv.std() * 0.3)

    if len(peaks) < 2:
        return max(3, len(profile) // 30), profile, peaks

    sorted_by_closeness = peaks[np.argsort(np.abs(peaks - center))]
    keep = np.sort(sorted_by_closeness[: min(8, len(peaks))])
    diffs = np.diff(keep)
    if len(diffs) == 0:
        return max(3, len(profile) // 30), profile, peaks

    spacing = float(np.median(diffs))
    length = int(round(len(profile) / max(spacing, 1e-6)))
    length = max(3, min(length, len(profile) // 2))
    return length, profile, peaks


def estimate_psf_spectrum(image: np.ndarray) -> dict:
    """Wrapper méthode 2."""
    angle, angles, scores = estimate_angle_spectrum(image)
    length, profile, troughs = estimate_length_from_profile(image, angle)
    return {
        "length": length,
        "angle_deg": angle,
        "angles": angles,
        "angle_scores": scores,
        "profile": profile,
        "troughs": troughs,
        "method": "spectrum",
    }


# ---------------------------------------------------------------------------
# API unifiée
# ---------------------------------------------------------------------------

def estimate_psf_params(image: np.ndarray, method: str = "cepstrum") -> dict:
    """
    method : "cepstrum" (défaut, robuste) ou "spectrum" (pédagogique).
    """
    if method == "cepstrum":
        return estimate_psf_cepstrum(image)
    elif method == "spectrum":
        return estimate_psf_spectrum(image)
    else:
        raise ValueError(f"méthode inconnue : {method}")
