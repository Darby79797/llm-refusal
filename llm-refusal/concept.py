from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field

import prompts

# --- Module-level defaults ---
DEFAULT_REFUSAL_TOKENS = [
    "I", "I'm", "As", "I cannot", "I'm sorry", "As an AI",
    " I", " I'm", " As", " I cannot", " I'm sorry", " As an AI"
]

DEFAULT_REFUSAL_PHRASES = [
    "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model",
    "As an assistant", "I cannot", "I can't", "I am unable to", "I'm unable to",
    "I am not able to", "I'm not able to"
]

DEFAULT_SEARCH_CONFIG = {
    "layer_cutoff_frac": 0.8,   # Search first 80% of layers
    "induce_threshold": 0,       # Induce score must be > 0
    "kl_threshold": 0.1,         # KL divergence must be < 0.1
}


@dataclass
class ConceptDefinition:
    """Defines a concept (e.g. refusal, sycophancy) for direction-finding experiments."""
    name: str                                                    # "refusal", "sycophancy", etc.
    train_data_fn: Callable[..., Tuple[List[str], List[str]]]   # () -> (positive, negative)
    eval_data_fn: Callable[..., Tuple[List[str], List[str]]]    # () -> (positive, negative)
    target_tokens: List[str]                                     # for LogOddsMetric during search
    detection_phrases: List[str]                                 # for string-match eval
    search_config: dict = field(default_factory=lambda: {        # search hyperparams
        "layer_cutoff_frac": 0.8,
        "induce_threshold": 0,
        "kl_threshold": 0.1,
    })


# --- Registry: maps string names to factory functions ---
CONCEPT_REGISTRY: Dict[str, Callable[[], ConceptDefinition]] = {}


def register_concept(name: str, factory: Callable[[], ConceptDefinition]):
    """Register a concept factory under a string name."""
    CONCEPT_REGISTRY[name] = factory


def get_concept(name: str) -> ConceptDefinition:
    """Look up a concept by string name. Raises KeyError if not registered."""
    return CONCEPT_REGISTRY[name]()


# --- Refusal concept (registered by default) ---
def make_refusal_concept() -> ConceptDefinition:
    return ConceptDefinition(
        name="refusal",
        train_data_fn=prompts.create_refusal_train_data,
        eval_data_fn=prompts.create_refusal_eval_data,
        target_tokens=DEFAULT_REFUSAL_TOKENS,
        detection_phrases=DEFAULT_REFUSAL_PHRASES,
        search_config=DEFAULT_SEARCH_CONFIG,
    )


register_concept("refusal", make_refusal_concept)
