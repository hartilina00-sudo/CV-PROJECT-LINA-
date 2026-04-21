"""
cnn_psf_estimator.py
--------------------
CNN (OPTIONNEL — uniquement autorisé pour ce projet selon la consigne)
qui prend le log-spectre FFT d'une image floue et prédit (L, θ).

⚠️ NOTE IMPORTANTE :
    D'après la consigne du projet, les CNN ne sont autorisés QUE pour
    l'estimation de la PSF (pas pour la déconvolution ni pour le reste
    du pipeline). Ce module est donc fourni à titre de démonstration
    complémentaire à la méthode du cepstre. La méthode PRINCIPALE du
    projet reste l'analyse fréquentielle non-apprise (cepstrum).

Le CNN ne fait qu'une régression à 2 sorties : [L, sin(2θ), cos(2θ)]
(on utilise 2θ à cause de la symétrie des PSFs rectilignes : θ et
θ+180° produisent la même PSF, donc on prédit sin/cos de 2θ pour
obtenir une sortie continue et bien définie).

Dépendances : torch (facultatif — le notebook fonctionne sans ce module).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class PSFRegressor(nn.Module):
        """Petit CNN de régression — prend un log-spectre 128x128."""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 3)   # [L_normalisé, sin(2θ), cos(2θ)]

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)


    def prepare_input(image: np.ndarray, target_size: int = 128) -> torch.Tensor:
        """
        Construit l'entrée du CNN : log-spectre centré, redimensionné
        à (target_size, target_size), normalisé dans [0,1].
        """
        from .psf_estimation import log_spectrum
        spec = log_spectrum(image)
        # redimension à target_size
        import cv2
        spec_resized = cv2.resize(spec, (target_size, target_size))
        spec_resized = (spec_resized - spec_resized.min()) / \
                       (spec_resized.max() - spec_resized.min() + 1e-8)
        t = torch.from_numpy(spec_resized).float()[None, None, :, :]
        return t


    def decode_output(pred: torch.Tensor,
                      length_scale: float = 30.0) -> tuple[float, float]:
        """Décodage : [L_norm, sin2θ, cos2θ] -> (L, θ°)."""
        p = pred.detach().cpu().numpy().flatten()
        L = max(1.0, float(p[0]) * length_scale)
        s2t, c2t = float(p[1]), float(p[2])
        angle = (np.degrees(np.arctan2(s2t, c2t)) / 2.0) % 180.0
        return L, angle


    def make_training_batch(n_samples: int = 64, size: int = 128,
                            length_scale: float = 30.0, seed: int | None = None):
        """
        Génère un batch synthétique à la volée pour entraîner le CNN.
        Retourne (X, Y) — spectres 1x128x128 et cibles [L_norm, s2θ, c2θ].
        """
        from .synthetic_data import generate_plate, apply_motion_blur, SAMPLE_PLATES
        rng = np.random.default_rng(seed)
        X = torch.zeros(n_samples, 1, size, size)
        Y = torch.zeros(n_samples, 3)

        for i in range(n_samples):
            text = SAMPLE_PLATES[rng.integers(0, len(SAMPLE_PLATES))]
            L = int(rng.integers(8, 28))
            theta = float(rng.uniform(0, 180))
            sharp = generate_plate(text=text, size=(128, 384))
            blurred, _ = apply_motion_blur(sharp, L, theta,
                                            noise_std=float(rng.uniform(0, 0.01)))
            X[i, 0] = prepare_input(blurred, size)[0, 0]
            Y[i, 0] = L / length_scale
            Y[i, 1] = float(np.sin(np.deg2rad(2 * theta)))
            Y[i, 2] = float(np.cos(np.deg2rad(2 * theta)))
        return X, Y


    def train_cnn(n_batches: int = 30, batch_size: int = 32,
                  device: str = "cpu", verbose: bool = True):
        """
        Entraîne rapidement le CNN sur des batches synthétiques à la volée.
        Retourne le modèle entraîné.
        """
        model = PSFRegressor().to(device)
        optim_ = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for step in range(n_batches):
            X, Y = make_training_batch(batch_size, seed=step)
            X, Y = X.to(device), Y.to(device)
            optim_.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            loss.backward()
            optim_.step()
            losses.append(loss.item())
            if verbose and (step % 5 == 0 or step == n_batches - 1):
                print(f"  step {step+1:3d}/{n_batches} | loss = {loss.item():.4f}")
        return model, losses
