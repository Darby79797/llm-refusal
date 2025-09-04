import pytest

import warnings

# Suppress Pydantic v1 validator warnings only for Hugging Face / Transformers libraries
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1 style `@validator` validators are deprecated.*",
    module=r"(transformers|huggingface_hub).*"
)

import torch as t
from unittest.mock import MagicMock, PropertyMock
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Import classes from your main script ---
# Note: To make this work, ensure your main script can be imported.
# You might need to add an __init__.py file in the root and adjust sys.path if necessary.
# For simplicity, we assume the classes are in a file named `main.py`.
from scratch import (
    PromptData,
    ChatPromptFormatter,
    ModelInterventionApplier,
    ActivationExtractor,
    DifferenceInMeans,
    Three_Score_Evaluator,
    DirectionVector,
    LogOddsMetric
)

# --- Pytest Configuration Hooks ---

def pytest_addoption(parser):
    """Adds a command-line option to pytest to run only smoke tests."""
    parser.addoption(
        "--run-smoke", action="store_true", default=False, help="run smoke tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modifies the test collection based on the --run-smoke flag."""
    if not config.getoption("--run-smoke"):
        # If --run-smoke is not given, skip tests marked with 'smoke'
        skip_smoke = pytest.mark.skip(reason="need --run-smoke option to run")
        for item in items:
            if "smoke" in item.keywords:
                item.add_marker(skip_smoke)
        return

    # If --run-smoke is given, only run tests marked with 'smoke'
    # and skip the rest.
    skip_non_smoke = pytest.mark.skip(reason="only running smoke tests")
    for item in items:
        if "smoke" not in item.keywords:
            item.add_marker(skip_non_smoke)

# --- Fixtures for Unit Tests (using Mocks) ---

@pytest.fixture(scope="module")
def mock_tokenizer():
    """Provides a mock tokenizer that simulates basic functionality."""
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.name_or_path = 'mock/gemma-tiny'
    tokenizer.model_max_length = 512
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.bos_token = "<bos>"

    tokenizer.encode = MagicMock()
    def mock_encode_func(text, add_special_tokens=False):
        # Return a deterministic, unique-ish integer list for any string.
        return [abs(hash(text)) % 50000]
    tokenizer.encode.side_effect = mock_encode_func
    
    # Simulate tokenization behavior
    def format_batch_side_effect(prompts, **kwargs):
        # A very basic tokenization simulation
        input_ids = [[0] * 10 for _ in prompts]
        attention_mask = [[1] * 10 for _ in prompts]
        return {
            'input_ids': t.tensor(input_ids, dtype=t.long),
            'attention_mask': t.tensor(attention_mask, dtype=t.long)
        }
    tokenizer.side_effect = format_batch_side_effect
    
    # Mock for ChatPromptFormatter specifically
    formatter_mock = MagicMock(spec=ChatPromptFormatter)
    formatter_mock.format_batch.side_effect = format_batch_side_effect
    
    return formatter_mock, tokenizer

@pytest.fixture(scope="module")
def mock_model():
    """Provides a mock model with a simplified structure."""
    model = MagicMock()
    
    # Mock config
    config = MagicMock()
    config.n_layer = 2
    config.n_head = 2
    type(model).config = PropertyMock(return_value=config)
    
    # Mock device
    type(model).device = PropertyMock(return_value=t.device("cpu"))

    # Mock layers for ModelInterventionApplier
    mock_layer = MagicMock()
    mock_layer.register_forward_hook.return_value = MagicMock()
    type(model).model = PropertyMock(return_value=MagicMock(layers=[mock_layer] * 2))

    # Mock call to return fake logits
    def model_side_effect(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        batch_size, seq_len = input_ids.shape
        mock_logits = t.randn(batch_size, seq_len, 50000) # vocab size
        return MagicMock(logits=mock_logits)
    model.return_value = model_side_effect
    
    return model

@pytest.fixture
def sample_prompt_data():
    """A small, balanced PromptData object for testing."""
    return PromptData(
        prompts=["pos 1", "pos 2", "neg 1", "neg 2"],
        labels=[True, True, False, False]
    )

@pytest.fixture
def mock_intervention_applier(mock_model):
    """Provides a mock ModelInterventionApplier."""
    return ModelInterventionApplier(mock_model)

# --- Fixtures for Smoke Tests (using a real, tiny model) ---

@pytest.fixture(scope="session")
def tiny_model_name():
    """The name of a small, fast model for smoke testing."""
    return "sshleifer/tiny-gpt2"

@pytest.fixture(scope="session")
def real_tiny_model_and_tokenizer(tiny_model_name):
    """Loads a real, small model and tokenizer once per test session."""
    try:
        model = AutoModelForCausalLM.from_pretrained(tiny_model_name)
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Failed to load tiny model for smoke tests: {e}")

