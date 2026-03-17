import json
import os
import torch as t
from typing import List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class PromptData:
    """A container for prompts and their corresponding labels."""
    # feels so unnecessary. Like, seriously. Is wrapping a list of prompts and bools in a dataclass worth it?
    prompts: List[str]
    labels: List[bool]  # True for positive examples, False for negative.

    def train_val_split(self, test_size: float = 0.2, random_state: int = 39):
        """Splits the data into training and validation sets."""
        train_prompts, val_prompts, train_labels, val_labels = train_test_split(
            self.prompts, self.labels, test_size=test_size, random_state=random_state, stratify=self.labels
        )
        return (
            PromptData(train_prompts, train_labels),
            PromptData(val_prompts, val_labels)
        )

@dataclass
class DirectionScores:
    """Holds three scores for evaluating a direction vector."""
    bypass: float  # Lower is better. We optimise on this score. Avg logodds metric on positive prompts with global ablation, testing how much we eliminate the behaviour.
    induce: float  # Higher is better. We satisfice on this score>0. Avg metric on negative prompts with layer-specific addition, testing sufficiency.
    kl: float      # Lower is better. We satisfice on this score<0.1. KL divergence on negative prompts with global ablation, testing if ablation nukes performance.
    induce_global: float = 0.0  # Like induce, but adds direction at all layers (Arditi-style). Default 0.0 for backward compat.

@dataclass
class DirectionVector:
    """Represents a direction vector found in the model's activation space."""
    vector: t.Tensor
    layer: int
    position_index: int  # A negative index, from the end of the prompt
    score: float  # The evaluation metric score for this direction

    @property
    def unit(self) -> t.Tensor:
        """Returns the unit vector of the direction."""
        return self.vector / t.norm(self.vector)

    def save(self, path: str) -> None:
        """Save to {path}.pt + {path}.json"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        t.save(self.vector, f"{path}.pt")
        meta = {"layer": self.layer, "position_index": self.position_index, "score": self.score}
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DirectionVector':
        """Load from {path}.pt + {path}.json"""
        vector = t.load(f"{path}.pt", weights_only=True)
        with open(f"{path}.json") as f:
            meta = json.load(f)
        return cls(vector=vector, layer=meta["layer"], position_index=meta["position_index"], score=meta["score"])
