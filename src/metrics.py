"""
metrics.py
----------
Métriques d'évaluation de la qualité de restauration.

PSNR et SSIM sont les deux métriques standard de la littérature en
déconvolution d'images. PSNR est directement dérivée de la MSE :

        MSE   = (1/N) Σ (x - x̂)²
        PSNR  = 10 · log10(MAX² / MSE)          [dB]

où MAX est la dynamique (255 pour uint8, 1.0 pour float). Plus le PSNR
est élevé, plus la restauration est proche de l'image de référence.
"""

from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Erreur quadratique moyenne entre deux images de même taille."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio, en dB. `data_range` = 255 pour uint8.
    """
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / m)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """
    SSIM version simple (une fenêtre globale, luminance + contraste + struct).
    Suffisant pour comparer quelques restaurations ; pour un SSIM pleinement
    localisé il faudrait skimage.metrics.structural_similarity, qu'on peut
    ajouter si disponible.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 3:
        # moyenne sur les canaux
        return float(np.mean([ssim(a[..., c], b[..., c], data_range)
                              for c in range(a.shape[2])]))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_a, mu_b = a.mean(), b.mean()
    va = a.var()
    vb = b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))

    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)
    return float(num / den) if den > 0 else 0.0
