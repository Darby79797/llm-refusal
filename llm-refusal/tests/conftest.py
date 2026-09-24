import os
import pytest

import warnings
import logging

# Must be set before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
# Bound the MPS caching allocator so it GCs / raises OOM instead of swapping
# (see batching.MPS_WATERMARKS; duplicated here because it must precede torch).
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

# Suppress Pydantic v1 validator warnings only for Hugging Face / Transformers libraries
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1 style `@validator` validators are deprecated.*",
    module=r"(transformers|huggingface_hub).*"
)

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress verbose logging from libraries for cleaner test output
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

from datatypes import PromptData

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
    # Ensure a pad token is set. Padding side must be RIGHT: the pipeline reads
    # the generation boundary as attention_mask.sum(-1)-1 and activations at
    # true_len+pos_idx, both of which are wrong under left padding. This fixture
    # previously set 'left', which is the configuration that produced the April
    # 2026 scoring bug. ChatPromptFormatter also re-asserts this per call.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    return model, tokenizer

