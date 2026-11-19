"""A small pure-NumPy MLP policy with a flat evolvable genome."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import ACTION_SIZE, LEGACY_OBSERVATION_SIZE, OBSERVATION_SIZE


HIDDEN_SIZES = (24, 24)
LAYER_SHAPES = (
    (OBSERVATION_SIZE, HIDDEN_SIZES[0]),
    (HIDDEN_SIZES[0],),
    (HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
    (HIDDEN_SIZES[1],),
    (HIDDEN_SIZES[1], ACTION_SIZE),
    (ACTION_SIZE,),
)
GENOME_SIZE = sum(int(np.prod(shape)) for shape in LAYER_SHAPES)
LEGACY_LAYER_SHAPES = (
    (LEGACY_OBSERVATION_SIZE, HIDDEN_SIZES[0]),
    (HIDDEN_SIZES[0],),
    (HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
    (HIDDEN_SIZES[1],),
    (HIDDEN_SIZES[1], ACTION_SIZE),
    (ACTION_SIZE,),
)
LEGACY_GENOME_SIZE = sum(int(np.prod(shape)) for shape in LEGACY_LAYER_SHAPES)


@dataclass
class MLPPolicy:
    """Observation-to-action network with tanh hidden/output activations."""

    genome: np.ndarray
    input_size: int = field(init=False, default=OBSERVATION_SIZE)

    def __post_init__(self) -> None:
        self.set_genome(self.genome)

    @classmethod
    def random(cls, rng: np.random.Generator) -> MLPPolicy:
        weights = []
        for in_size, out_size in (
            (OBSERVATION_SIZE, HIDDEN_SIZES[0]),
            (HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            (HIDDEN_SIZES[1], ACTION_SIZE),
        ):
            scale = np.sqrt(2.0 / (in_size + out_size))
            weights.append(rng.normal(0.0, scale, size=(in_size, out_size)))
        genome = np.concatenate(
            [weights[0].ravel(), np.zeros(HIDDEN_SIZES[0]), weights[1].ravel(),
             np.zeros(HIDDEN_SIZES[1]), weights[2].ravel(), np.zeros(ACTION_SIZE)]
        )
        return cls(genome.astype(float, copy=False))

    @staticmethod
    def genome_length() -> int:
        return GENOME_SIZE

    def get_genome(self) -> np.ndarray:
        return self.genome.copy()

    def set_genome(self, genome: np.ndarray) -> None:
        flat = np.asarray(genome, dtype=float).reshape(-1)
        if flat.shape == (GENOME_SIZE,):
            self.input_size = OBSERVATION_SIZE
        elif flat.shape == (LEGACY_GENOME_SIZE,):
            self.input_size = LEGACY_OBSERVATION_SIZE
        else:
            raise ValueError(
                f"genome must contain {LEGACY_GENOME_SIZE} (v1) or {GENOME_SIZE} (v2) values"
            )
        if not np.all(np.isfinite(flat)):
            raise ValueError("genome values must be finite")
        self.genome = flat.copy()

    def _parameters(self):
        offset = 0
        parameters = []
        shapes = LAYER_SHAPES if self.input_size == OBSERVATION_SIZE else LEGACY_LAYER_SHAPES
        for shape in shapes:
            count = int(np.prod(shape))
            parameters.append(self.genome[offset : offset + count].reshape(shape))
            offset += count
        return parameters

    def trace(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return hidden activations and pre-tanh output values for inspection."""
        obs = np.asarray(observation, dtype=float)
        if obs.shape == (LEGACY_OBSERVATION_SIZE,) and self.input_size == OBSERVATION_SIZE:
            obs = np.pad(obs, (0, 1))
        elif obs.shape == (OBSERVATION_SIZE,) and self.input_size == LEGACY_OBSERVATION_SIZE:
            obs = obs[:LEGACY_OBSERVATION_SIZE]
        if obs.shape != (self.input_size,):
            raise ValueError(f"observation must have shape ({self.input_size},) for policy schema")
        w1, b1, w2, b2, w3, b3 = self._parameters()
        hidden1 = np.tanh(obs @ w1 + b1)
        hidden2 = np.tanh(hidden1 @ w2 + b2)
        raw_outputs = hidden2 @ w3 + b3
        return hidden1, hidden2, raw_outputs

    def act(self, observation: np.ndarray) -> np.ndarray:
        _, _, raw_outputs = self.trace(observation)
        return np.tanh(raw_outputs)


def get_genome(policy: MLPPolicy) -> np.ndarray:
    """Convenience genome getter for generic evolutionary code."""
    return policy.get_genome()


def set_genome(policy: MLPPolicy, genome: np.ndarray) -> None:
    """Convenience genome setter for generic evolutionary code."""
    policy.set_genome(genome)

