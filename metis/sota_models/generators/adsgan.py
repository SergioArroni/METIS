"""ADS-GAN (Yoon, Drumright & van der Schaar, 2020) — not implemented.

The previous class was a placeholder that trained CTGAN under the ADS-GAN
name, with only a warning: anyone asking for ADS-GAN silently got CTGAN.
ADS-GAN adds an identifiability penalty to the generator loss (synthetic
records must not be closer to a real record than its nearest real neighbour)
and needs its own training loop; no config of the paper uses it.  It is not
registered, and building it raises, so the name can no longer hide CTGAN.
"""

from .base import BaseGenerator


class ADSGANGenerator(BaseGenerator):
    """Placeholder that refuses to run (module docstring)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ADS-GAN is not implemented in METIS (it used to train CTGAN under this "
            "name); use 'ctgan' explicitly or add a real ADS-GAN implementation"
        )

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - unreachable
        raise NotImplementedError

    def generate(self, n_samples: int):  # pragma: no cover - unreachable
        raise NotImplementedError
