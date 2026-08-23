"""Synthetic tiling screens, for demos and tests only.

This is scaffolding, not science: it generates data with the *structure* of a
real Cas13 tiling screen -- overlapping windows walked along a handful of
transcripts, a per-transcript baseline offset, measurement noise -- so the
training harness can be exercised and its failure modes demonstrated on data
whose ground truth is known.

The per-transcript offset is the important part. Real transcripts differ in
accessibility, abundance and local context, so two windows from the same
transcript are correlated beyond their shared sequence. That correlation is
exactly what an ungrouped train/test split leaks, and simulating it is what
lets :mod:`cas13guide.services.training` prove the leak rather than assert it.
"""

from __future__ import annotations

import numpy as np

from .models.features import FeatureSpec
from .services.training import TrainingExample
from .util.duplex import profile_duplex
from .util.seq import revcomp

BASES = "AUGC"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z)) if z >= 0 else float(np.exp(z) / (1.0 + np.exp(z)))


def simulate_screen(
    n_transcripts: int = 12,
    transcript_length: int = 240,
    spacer_length: int = 22,
    step: int = 3,
    mismatch_rate: float = 0.35,
    transcript_effect_sd: float = 1.2,
    noise_sd: float = 0.25,
    signal_scale: float = 1.6,
    seed: int = 0,
    effector: str = "RfxCas13d",
    assay: str = "cell_culture_knockdown",
) -> tuple[list[TrainingExample], np.ndarray]:
    """Generate a screen and return ``(examples, true_weights)``.

    Activity is ``sigmoid(w . x + transcript_effect + noise)`` where ``x`` is
    the real feature vector, so a model that recovers ``w`` is learning the
    intended signal and nothing else.
    """
    rng = np.random.default_rng(seed)
    spec = FeatureSpec.for_length(spacer_length)

    # A sparse true weight vector: most features do nothing, as in reality.
    true_w = np.zeros(spec.n_features)
    active = rng.choice(spec.n_features, size=spec.n_features // 6, replace=False)
    true_w[active] = rng.normal(0.0, 1.0, size=active.size)

    examples: list[TrainingExample] = []
    for t in range(n_transcripts):
        sequence = "".join(rng.choice(list(BASES), size=transcript_length))
        offset = float(rng.normal(0.0, transcript_effect_sd))
        group = f"ENST{t:05d}"

        for start in range(0, transcript_length - spacer_length + 1, step):
            site = list(sequence[start : start + spacer_length])
            spacer = revcomp("".join(site))

            # Some rows are off-target-like: perturb the observed site.
            if rng.random() < mismatch_rate:
                for _ in range(int(rng.integers(1, 5))):
                    i = int(rng.integers(0, spacer_length))
                    site[i] = str(rng.choice([b for b in BASES if b != site[i]]))
            observed_site = "".join(site)

            duplex = profile_duplex(spacer, observed_site)
            x = spec.transform(spacer, duplex, None)
            z = signal_scale * float(x @ true_w) / np.sqrt(spec.n_features)
            z += offset + float(rng.normal(0.0, noise_sd))

            examples.append(
                TrainingExample(
                    spacer=spacer,
                    target_site=observed_site,
                    observed_activity=min(1.0, max(0.0, _sigmoid(z))),
                    group=group,
                    effector=effector,
                    assay=assay,
                )
            )

    return examples, true_w
