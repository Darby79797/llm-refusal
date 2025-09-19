import pytest

import warnings
import logging

# Suppress Pydantic v1 validator warnings only for Hugging Face / Transformers libraries
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1 style `@validator` validators are deprecated.*",
    module=r"(transformers|huggingface_hub).*"
)

import torch as t
from unittest.mock import MagicMock, PropertyMock
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress verbose logging from libraries for cleaner test output
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("scratch").setLevel(logging.WARNING)

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

@pytest.fixture
def mock_model():
    """A more realistic mock for AutoModelForCausalLM."""
    mock = MagicMock(spec=AutoModelForCausalLM)
    
    # FIX: Add .dtype and .device attributes
    mock.dtype = t.float32
    mock.device = t.device("cpu")
    
    # FIX: Correctly mock the nested .model.layers structure
    mock.model = MagicMock()
    mock_layer = MagicMock()
    # Ensure the hook's remove() method is also a mock so it can be called
    mock_layer.register_forward_hook.return_value = MagicMock(remove=MagicMock())
    mock.model.layers = [mock_layer] 
    
    # Add a minimal config that the code expects, including the existence of a mock.config
    mock.config = MagicMock()
    mock.config.hidden_size = 10 
    
    return mock

@pytest.fixture
def mock_intervention_applier(mock_model):
    """A mock for ModelInterventionApplier that includes the .model attribute."""
    # FIX: The mock now correctly holds a reference to the mock_model
    mock_applier = MagicMock(spec=ModelInterventionApplier)
    mock_applier.model = mock_model
    mock_applier.transformer_layers = mock_model.model.layers
    return mock_applier

@pytest.fixture(params=[True, False], ids=["with_chat_template", "without_chat_template"])
def mock_tokenizer(request):
    """
    A parameterized fixture to mock AutoTokenizer with all necessary methods.
    """
    has_chat_template = request.param
    mock = MagicMock(spec=AutoTokenizer)
    
    # FIX: Define all methods that could be called by ChatPromptFormatter
    mock.apply_chat_template = MagicMock(return_value=t.tensor([[1, 2, 3]]))
    mock.__call__ = MagicMock(return_value={'input_ids': t.tensor([[1, 2, 3]]), 'attention_mask': t.tensor([[1, 1, 1]])})
    mock.encode = MagicMock(side_effect=lambda token, **kwargs: [hash(token) % 50000])

    # Use PropertyMock to control the presence of .chat_template
    if has_chat_template:
        type(mock).chat_template = PropertyMock(return_value="A template string")
    else:
        type(mock).chat_template = PropertyMock(return_value=None)

    # Add other necessary attributes
    mock.pad_token = "[PAD]"
    mock.pad_token_id = hash(mock.pad_token) % 50000
    mock.padding_side = 'left'
    mock.model_max_length = 4096
    
    return mock

@pytest.fixture
def sample_prompt_data():
    """A small, balanced PromptData object for testing."""
    return PromptData(
        prompts=["pos 1", "pos 2", "neg 1", "neg 2"],
        labels=[True, True, False, False]
    )

# --- Fixtures for Smoke Tests (using a real, tiny model) ---

@pytest.fixture(scope="session")
def tiny_model_name():
    """The name of a small, fast model for smoke testing."""
    return "google/gemma-3-270m" # this model is not tiny - some of our tests rely on minmal model intelligence, so we scale up for now.
    # return "sshleifer/tiny-gpt2"

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


# --- Fixtures for generation tests ---
#GENERATION_TEST_MODELS = ["gpt2", "roneneldan/TinyStories-1M", "google/gemma-3-270m"]#, "Qwen/Qwen1.5-1.8B"] # add gemma, qwen, llama, etc
GENERATION_TEST_MODELS = [
    "google/gemma-3-1b-pt",
    "google/gemma-3-1b-it", # Expected to fail. Fails
    #"Qwen/Qwen1.5-1.8B-Chat", # Expected to pass, but doesn't
    #"openai-community/gpt2",
    "openai-community/gpt2-xl",
]
#GENERATION_TEST_MODELS = ["Qwen/Qwen1.5-1.8B"]
@pytest.fixture(scope="session", params=GENERATION_TEST_MODELS)
def model_and_tokenizer(request):
    """
    Pytest fixture to load a model and tokenizer once per session.
    Parametrized to run tests across all models in GENERATION_TEST_MODELS.
    """
    model_name = request.param

    if t.cuda.is_available():
        device = t.device("cuda")
    elif t.backends.mps.is_available():
        device = t.device("mps")
    else:
        device = t.device("cpu")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.fail(f"Failed to load model or tokenizer for {model_name}: {e}")

    # --- Critical for Batching ---
    # Ensure a pad token is set, and padding side is left for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer

