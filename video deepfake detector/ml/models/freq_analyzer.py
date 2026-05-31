"""
Frequency Domain Deepfake Analyzer
====================================
GAN generators and upsampling operations leave invisible traces in
the frequency domain that CNNs operating in pixel space often miss.

Key phenomena exploited:
  1. GAN checkerboard artifacts → grid peaks in FFT magnitude spectrum
  2. Upsampling aliasing → periodic high-frequency patterns
  3. Compression-inconsistent regions → mismatched DCT coefficient distributions

Pipeline:
  RGB face crop
  → Greyscale  →  2D FFT  →  FFT-shift  →  log-magnitude  [B, 1, H, W]
  → 5-layer compact CNN
  → Sigmoid score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FrequencyAnalyzer(nn.Module):
    """
    Frequency-domain artifact detector via FFT + compact CNN.

    Detects:
      - GAN checkerboard patterns (transposed-conv overlap artefacts)
      - Upsampling aliasing from face decoder networks
      - DCT inconsistencies between spliced face region and background
    """

    def __init__(self):
        super().__init__()

        logger.info("Initialising FrequencyAnalyzer")

        # ── Compact CNN on log-magnitude FFT spectrum ─────────────────────────
        self.freq_cnn = nn.Sequential(
            # Block 1 — 1→32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # H/2, W/2

            # Block 2 — 32→64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # H/4, W/4

            # Block 3 — 64→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # H/8, W/8

            # Block 4 — 128→256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # → [B, 256, 4, 4]

            nn.Flatten(),             # [B, 4096]
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    # ── FFT Computation ───────────────────────────────────────────────────────

    def compute_fft(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB image to log-magnitude FFT spectrum.

        Args:
            img_tensor: [B, 3, H, W]  — face crop, pixel values [0,1]

        Returns:
            magnitude: [B, 1, H, W]  — log-magnitude spectrum, normalised
        """
        # Convert to greyscale: weighted luminance
        gray = (
            0.299 * img_tensor[:, 0:1, :, :]
            + 0.587 * img_tensor[:, 1:2, :, :]
            + 0.114 * img_tensor[:, 2:3, :, :]
        )  # [B, 1, H, W]

        # 2D FFT on spatial dims
        fft      = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)  # Zero-freq to centre

        # Log-magnitude (adds 1e-8 for numerical stability)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)

        # Normalise to [0, 1] per image
        mn = magnitude.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        mx = magnitude.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        magnitude = (magnitude - mn) / (mx - mn + 1e-8)

        return magnitude  # [B, 1, H, W]

    def compute_dct_inconsistency(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Approximate DCT-domain inconsistency between image quadrants.
        High variance between quadrant DCT stats → possible splicing.

        Returns: [B, 1] — inconsistency score
        """
        B, C, H, W = img_tensor.shape
        gray = img_tensor.mean(dim=1)  # [B, H, W]

        # Split into 4 quadrants
        h2, w2 = H // 2, W // 2
        quads = [
            gray[:, :h2, :w2],
            gray[:, :h2, w2:],
            gray[:, h2:, :w2],
            gray[:, h2:, w2:],
        ]

        # Compute mean absolute value of 2D FFT (proxy for DCT energy) per quadrant
        energies = []
        for q in quads:
            fft_q = torch.fft.fft2(q)
            energy = torch.abs(fft_q).mean(dim=(-2, -1))  # [B]
            energies.append(energy)

        energies = torch.stack(energies, dim=1)  # [B, 4]
        # High std across quadrants → inconsistency
        inconsistency = energies.std(dim=1, keepdim=True)  # [B, 1]
        # Normalise with sigmoid
        return torch.sigmoid(inconsistency - inconsistency.mean())

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] — face crops

        Returns:
            score: [B, 1] — fake probability (frequency domain)
        """
        freq_map = self.compute_fft(x)           # [B, 1, H, W]
        features = self.freq_cnn(freq_map)        # [B, 4096]
        score    = self.classifier(features)      # [B, 1]
        return score

    def get_frequency_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw FFT magnitude map for visualisation."""
        return self.compute_fft(x)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrequencyAnalyzer().to(device)
    dummy = torch.rand(4, 3, 224, 224).to(device)

    score = model(dummy)
    dct   = model.compute_dct_inconsistency(dummy)
    print(f"✅ FrequencyAnalyzer | score: {score.shape} | dct: {dct.shape}")
    # Expected: score [4,1], dct [4,1]
