import torch as t
from typing import Dict, Tuple
from abc import ABC, abstractmethod
import logging

from activations import ActivationExtractor
from datatypes import PromptData

logger = logging.getLogger(__name__)


class DirectionMethod(ABC):
    """Abstract base class for methods that compute direction vectors from data."""
    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor

    @abstractmethod
    def compute_difference_vectors(
        self,
        train_data: PromptData,
        max_positions: int = 5
    ) -> Dict[Tuple[int, int], t.Tensor]:
        ...


class DifferenceInMeans(DirectionMethod):
    """Calculates the difference-in-means direction vector between positive and negative prompt activations."""

    def compute_difference_vectors(
        self,
        train_data: PromptData,
        max_positions: int = 5
    ) -> Dict[Tuple[int, int], t.Tensor]:
        """
        Computes the difference-in-means vectors for all layer and position combinations.
        """
        positive_prompts = train_data.positive
        negative_prompts = train_data.negative

        logger.info(f"Computing activations for {len(positive_prompts)} positive and {len(negative_prompts)} negative prompts.")

        pos_activations = self.extractor.extract_residual_activations(positive_prompts, max_positions)
        neg_activations = self.extractor.extract_residual_activations(negative_prompts, max_positions)

        # Subtraction in float64 (from ActivationExtractor), then cast to float32 for storage/GPU
        difference_vectors = {
            key: (pos_activations[key] - neg_activations[key]).to(t.float32)
            for key in pos_activations if key in neg_activations
        }

        return difference_vectors
