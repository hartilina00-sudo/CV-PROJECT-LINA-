"""
wiener.py
---------
Restauration d'images floutées par déconvolution fréquentielle.

Modèle de dégradation étudié en cours :
        y = h * x + n
où * est la convolution 2D, h la PSF, n un bruit.

Le filtre de Wiener fournit l'estimateur linéaire qui minimise
l'erreur quadratique moyenne entre x et x_hat. Dans le domaine
de Fourier il s'écrit :

                     H*(u,v)
        X̂(u,v) = ─────────────── · Y(u,v)
                |H(u,v)|² + K(u,v)

où K(u,v) = S_n(u,v) / S_x(u,v) est le ratio densité spectrale
de bruit / densité spectrale du signal. En pratique on approxime
K par une CONSTANTE (paramètre de régularisation), ce qui donne
la forme utilisée dans ce projet :

                     H*(u,v)
        X̂(u,v) = ─────────────── · Y(u,v)
                |H(u,v)|² + K
"""

from __future__ import annotations

import numpy as np


def _pad_psf_to(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad + roll de la PSF pour que son centre soit à l'origine (0,0)."""
    H, W = shape
    kh, kw = psf.shape
    padded = np.zeros((H, W), dtype=np.float64)
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -kh // 2 + 1, axis=0)
    padded = np.roll(padded, -kw // 2 + 1, axis=1)
    return padded


def wiener_deconvolve_channel(y: np.ndarray, psf: np.ndarray,
                              K: float = 0.01) -> np.ndarray:
    """
    Wiener sur UN canal (image 2D float dans [0,1]).
    """
    H, W = y.shape
    psf_padded = _pad_psf_to(psf, (H, W))

    Y = np.fft.fft2(y)
    H_otf = np.fft.fft2(psf_padded)

    H_conj = np.conj(H_otf)
    denom = np.abs(H_otf) ** 2 + K
    W_filter = H_conj / denom

    X_hat = W_filter * Y
    x_hat = np.real(np.fft.ifft2(X_hat))
    return x_hat


def wiener_deconvolve(image: np.ndarray, psf: np.ndarray,
                      K: float = 0.01) -> np.ndarray:
    """
    Filtre de Wiener sur une image entière (gris ou RGB/BGR).

    Paramètres
    ----------
    image : np.ndarray uint8 ou float
    psf   : np.ndarray 2D, somme = 1
    K     : float > 0, régularisation (plus K grand, plus on lisse)

    Retour
    ------
    np.ndarray uint8, même forme que `image`.
    """
    if image.dtype == np.uint8:
        y = image.astype(np.float64) / 255.0
    else:
        y = image.astype(np.float64)

    if y.ndim == 2:
        x_hat = wiener_deconvolve_channel(y, psf, K)
    elif y.ndim == 3:
        x_hat = np.empty_like(y)
        for c in range(y.shape[2]):
            x_hat[..., c] = wiener_deconvolve_channel(y[..., c], psf, K)
    else:
        raise ValueError("image doit être 2D ou 3D")

    x_hat = np.clip(x_hat, 0.0, 1.0)
    return (x_hat * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pour comparaison : inverse filter (pas régularisé) et déconvolution
# d'un noyau Gaussien (filtre de base qui sert à comparer).
# ---------------------------------------------------------------------------

def inverse_filter_channel(y: np.ndarray, psf: np.ndarray,
                           eps: float = 1e-3) -> np.ndarray:
    """Filtre inverse brut : X̂ = Y / H (avec un epsilon pour éviter 1/0)."""
    H, W = y.shape
    psf_padded = _pad_psf_to(psf, (H, W))
    Y = np.fft.fft2(y)
    H_otf = np.fft.fft2(psf_padded)
    H_safe = np.where(np.abs(H_otf) < eps, eps, H_otf)
    return np.real(np.fft.ifft2(Y / H_safe))


def inverse_filter(image: np.ndarray, psf: np.ndarray,
                   eps: float = 1e-3) -> np.ndarray:
    """Filtre inverse sur image complète."""
    if image.dtype == np.uint8:
        y = image.astype(np.float64) / 255.0
    else:
        y = image.astype(np.float64)

    if y.ndim == 2:
        x_hat = inverse_filter_channel(y, psf, eps)
    else:
        x_hat = np.empty_like(y)
        for c in range(y.shape[2]):
            x_hat[..., c] = inverse_filter_channel(y[..., c], psf, eps)
    x_hat = np.clip(x_hat, 0.0, 1.0)
    return (x_hat * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Recherche automatique du K optimal
# ---------------------------------------------------------------------------

def find_best_K(blurred: np.ndarray, reference: np.ndarray, psf: np.ndarray,
                K_grid: np.ndarray | None = None) -> tuple[float, float, list]:
    """
    Balaye plusieurs valeurs de K et retourne celle qui maximise le PSNR
    vis-à-vis d'une image de référence (utile quand on a le ground truth).

    Retour
    ------
    best_K, best_psnr, history (liste de (K, psnr))
    """
    from .metrics import psnr
    if K_grid is None:
        K_grid = np.logspace(-4, 0, 25)   # 1e-4 ... 1.0
    history = []
    best_K, best_psnr = K_grid[0], -np.inf
    for K in K_grid:
        restored = wiener_deconvolve(blurred, psf, K=K)
        p = psnr(reference, restored)
        history.append((float(K), float(p)))
        if p > best_psnr:
            best_psnr = p
            best_K = float(K)
    return best_K, best_psnr, history
