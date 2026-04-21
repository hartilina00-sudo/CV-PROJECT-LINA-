"""
synthetic_data.py
-----------------
Génération d'images synthétiques de plaques d'immatriculation
et application d'un flou de mouvement avec PSF connue.

Toutes les techniques utilisées ici viennent du matériel de cours :
  - convolution 2D via FFT
  - PSF (Point Spread Function) linéaire paramétrée par (longueur, angle)
  - ajout de bruit gaussien pour simuler des conditions réelles
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple


# ---------------------------------------------------------------------------
# Génération de plaques synthétiques
# ---------------------------------------------------------------------------

def generate_plate(text: str = "123-ABC-45",
                   size: Tuple[int, int] = (128, 384),
                   bg_color: Tuple[int, int, int] = (255, 255, 255),
                   fg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Construit une image de plaque d'immatriculation synthétique.

    Paramètres
    ----------
    text : str
        Le texte à écrire sur la plaque.
    size : (H, W)
        Dimensions de l'image en pixels (hauteur, largeur).
    bg_color, fg_color : BGR
        Couleurs de fond et d'avant-plan (OpenCV travaille en BGR).

    Retour
    ------
    np.ndarray uint8 de forme (H, W, 3)
    """
    H, W = size
    img = np.full((H, W, 3), bg_color, dtype=np.uint8)

    # Bordure noire pour simuler le cadre d'une plaque
    cv2.rectangle(img, (4, 4), (W - 5, H - 5), (0, 0, 0), thickness=3)

    # Choix de la taille de police en fonction de la largeur
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = W / 260.0
    thickness = max(2, int(W / 150))

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (W - tw) // 2
    y = (H + th) // 2

    cv2.putText(img, text, (x, y), font, font_scale, fg_color,
                thickness, lineType=cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# PSF de flou de mouvement linéaire
# ---------------------------------------------------------------------------

def motion_blur_kernel(length: int, angle_deg: float,
                       ksize: int | None = None) -> np.ndarray:
    """
    Construit une PSF de flou de mouvement rectiligne.

    C'est un segment de pixels de longueur `length` orienté à `angle_deg`
    degrés par rapport à l'horizontale, normalisé pour que sa somme vaille 1
    (ainsi la convolution préserve la luminance moyenne de l'image).

    Paramètres
    ----------
    length : int
        Nombre de pixels sur lesquels le mouvement s'est produit.
    angle_deg : float
        Angle en degrés. 0 = flou horizontal, 90 = flou vertical.
    ksize : int, optionnel
        Taille du support du noyau (carré). Par défaut : length*2+1 pour
        laisser de la marge quel que soit l'angle.

    Retour
    ------
    np.ndarray float64 de forme (ksize, ksize), somme = 1.
    """
    if length < 1:
        raise ValueError("length doit être >= 1")
    if ksize is None:
        ksize = length * 2 + 1
    if ksize % 2 == 0:
        ksize += 1   # on veut un noyau de taille impaire, centre bien défini

    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    center = ksize // 2

    # On trace un segment de (center, center) dans la direction `angle_deg`
    # et on marque les pixels traversés.
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = -np.sin(theta)  # y va vers le bas dans une image, d'où le signe

    # On échantillonne avec un pas fin puis on arrondit les positions.
    # Pas = 0.5 pixel pour être sûr de ne pas rater des pixels.
    n_samples = int(length * 4) + 1
    ts = np.linspace(-length / 2, length / 2, n_samples)
    xs = np.round(center + ts * dx).astype(int)
    ys = np.round(center + ts * dy).astype(int)

    for x, y in zip(xs, ys):
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0

    s = kernel.sum()
    if s == 0:
        # fallback : delta si la longueur est trop courte
        kernel[center, center] = 1.0
    else:
        kernel /= s
    return kernel


# ---------------------------------------------------------------------------
# Convolution via FFT (dégradation y = h * x + n)
# ---------------------------------------------------------------------------

def _pad_psf_to(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Place la PSF dans une image de taille `shape` en la centrant sur (0,0)
    (convention FFT circulaire), pour que FFT(psf_padded) soit l'OTF.
    """
    H, W = shape
    kh, kw = psf.shape
    padded = np.zeros((H, W), dtype=np.float64)
    padded[:kh, :kw] = psf
    # On recentre : le centre de la PSF doit être à l'origine (0,0)
    padded = np.roll(padded, -kh // 2 + 1, axis=0)
    padded = np.roll(padded, -kw // 2 + 1, axis=1)
    return padded


def convolve_fft(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolution circulaire 2D via FFT. Fonctionne en niveaux de gris
    ou en RGB (canal par canal).
    """
    if image.ndim == 2:
        H, W = image.shape
        psf_padded = _pad_psf_to(psf, (H, W))
        F = np.fft.fft2(image)
        H_otf = np.fft.fft2(psf_padded)
        return np.real(np.fft.ifft2(F * H_otf))
    elif image.ndim == 3:
        out = np.empty_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            out[..., c] = convolve_fft(image[..., c], psf)
        return out
    else:
        raise ValueError("image doit être 2D ou 3D")


def apply_motion_blur(image: np.ndarray, length: int, angle_deg: float,
                      noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simule l'équation de dégradation standard :
         y = H x + n
    où H est la convolution par une PSF de flou de mouvement et n un
    bruit gaussien.

    Paramètres
    ----------
    image : np.ndarray uint8 ou float in [0,1]
    length, angle_deg : paramètres de la PSF
    noise_std : écart-type du bruit gaussien en fraction de 1.0
                (ex. 0.01 = bruit modéré)

    Retour
    ------
    (blurred_uint8, psf)  — l'image floue et la PSF utilisée.
    """
    # Normalisation vers [0,1]
    if image.dtype == np.uint8:
        x = image.astype(np.float64) / 255.0
    else:
        x = image.astype(np.float64)

    psf = motion_blur_kernel(length, angle_deg)
    y = convolve_fft(x, psf)

    if noise_std > 0:
        y = y + np.random.normal(0, noise_std, y.shape)

    y = np.clip(y, 0.0, 1.0)
    return (y * 255).astype(np.uint8), psf


# ---------------------------------------------------------------------------
# Génération d'un dataset
# ---------------------------------------------------------------------------

# Petit jeu de textes variés qui imitent des formats de plaques internationaux
SAMPLE_PLATES = [
    "123-ABC-45", "AB-123-CD", "7890-XY", "MA-2024-RB",
    "CAS-4578", "1-A-1234", "EU-999-OK", "TGV-2025",
    "X4K9-Z7", "BERLIN-42", "NYC-007", "UIR-2024",
    "ML-AI-99", "DL-CNN-1", "IMG-512-X", "JPG-RAW-0",
    "SPEED-88", "RACE-LMP", "AUTO-ABC", "PLATE-123",
]


def build_dataset(sharp_dir: str, blurred_dir: str,
                  n_samples: int = 40,
                  size: Tuple[int, int] = (128, 384),
                  seed: int = 42,
                  noise_std: float = 0.005) -> list[dict]:
    """
    Crée un dataset complet d'images nettes + floues et sauvegarde sur disque.

    Chaque exemple reçoit une PSF tirée aléatoirement :
        length ∈ [9, 25]  pixels
        angle  ∈ [0, 180) degrés

    Retour
    ------
    Liste de dictionnaires {id, text, length, angle, sharp_path, blurred_path}
    """
    import os
    rng = np.random.default_rng(seed)
    metadata = []

    for i in range(n_samples):
        text = SAMPLE_PLATES[i % len(SAMPLE_PLATES)]
        sharp = generate_plate(text=text, size=size)
        length = int(rng.integers(9, 26))        # 9..25
        angle = float(rng.uniform(0, 180))       # 0..180

        blurred, _psf = apply_motion_blur(sharp, length, angle,
                                          noise_std=noise_std)

        sharp_path = os.path.join(sharp_dir, f"plate_{i:03d}.png")
        blurred_path = os.path.join(blurred_dir, f"plate_{i:03d}.png")
        cv2.imwrite(sharp_path, sharp)
        cv2.imwrite(blurred_path, blurred)

        metadata.append({
            "id": i, "text": text,
            "length": length, "angle": angle,
            "sharp_path": sharp_path, "blurred_path": blurred_path,
        })
    return metadata
