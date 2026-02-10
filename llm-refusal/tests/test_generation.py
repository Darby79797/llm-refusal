import pytest
import torch as t
import warnings
from scratch import ( # Replace with your actual module name
    generate_with_hooks, 
    ChatPromptFormatter, 
    ModelInterventionApplier,
    InterventionSuite,
    DirectionVector
)

# --- Test Scenarios ---
# We define different prompt configurations to test against.
TEST_PROMPTS = {
    "single_prompt": ["A simple test prompt."],
    "batch_same_length": ["First test prompt.", "Second test vector."],
    "batch_diff_length": ["A short one.", "This is a much longer one to check padding."]
}

# A dummy hook that performs an identity operation. Its purpose is to test
# if the presence of hooks disrupts the generation logic.
def id_hook(module, input, output):
    return output

def test_chat_prompt_formatter_padding(model_and_tokenizer):
    """
    Tests that the ChatPromptFormatter correctly handles batches with prompts of different lengths.
    """
    model, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer)

    prompts = [
        "This is a short prompt.",
        "This is a significantly longer prompt to test the padding logic."
    ]

    batch = formatter.format_batch(prompts)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # --- Assertions ---
    # Check for correct batch size and tensor type
    assert input_ids.shape[0] == 2
    assert isinstance(input_ids, t.Tensor)
    
    # Check that padding was applied correctly (left padding)
    assert input_ids.shape[1] > tokenizer(prompts[0], return_tensors='pt')['input_ids'].shape[1]

    if formatter.prepend_bos:
        # Base models: position 0 is BOS (attended) for ALL rows, padding starts at position 1
        assert input_ids[0, 0].item() == tokenizer.bos_token_id, "First token of shorter prompt should be BOS."
        assert input_ids[1, 0].item() == tokenizer.bos_token_id, "First token of longer prompt should be BOS."
        assert attention_mask[0, 0].item() == 1, "BOS token should be attended to."
        assert attention_mask[0, 1].item() == 0, "After BOS, shorter prompt should have padding."
        assert attention_mask[1, 0].item() == 1, "BOS token of longer prompt should be attended to."
    else:
        # Instruction-tuned models: no BOS prepended, padding is at the start
        assert attention_mask[0, 0].item() == 0, "The first token of the shorter prompt should be masked."
        assert attention_mask[1, 0].item() == 1, "The first token of the longer prompt should not be masked."

    # Check that the last token is never a pad token
    assert input_ids[0, -1].item() != tokenizer.pad_token_id
    assert input_ids[1, -1].item() != tokenizer.pad_token_id

@pytest.mark.parametrize("prompts", TEST_PROMPTS.values(), ids=TEST_PROMPTS.keys())
def test_generate_with_hooks_logic(model_and_tokenizer, prompts):
    """
    Tests the manual `generate_with_hooks` function for correctness with and without hooks.
    """
    model, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer)
    
    # 1. Get baseline output (no hooks)
    baseline_output = generate_with_hooks(model, tokenizer, formatter, prompts, max_new_tokens=10)
    
    assert isinstance(baseline_output, list)
    assert len(baseline_output) == len(prompts)
    assert all(isinstance(s, str) for s in baseline_output)

    # 2. Get output with a non-interfering hook on every layer
    intervention_applier = ModelInterventionApplier(model)
    hooks = [
        layer.register_forward_hook(id_hook) 
        for layer in intervention_applier.transformer_layers
    ]
    
    hooked_output = generate_with_hooks(model, tokenizer, formatter, prompts, max_new_tokens=10)
    
    # --- Critical ---
    # Cleanup: Always remove hooks after the test.
    for hook in hooks:
        hook.remove()
        
    # 3. Assert that the outputs are identical
    assert baseline_output == hooked_output, \
        "Generation output changed when non-interfering hooks were added."

@pytest.mark.parametrize("prompts", TEST_PROMPTS.values(), ids=TEST_PROMPTS.keys())
def test_intervention_suite_generation_logic(model_and_tokenizer, prompts):
    """
    Tests the InterventionSuite's `.test_generation` method for correctness.
    It verifies that a zero-strength ablation produces the same output as the baseline.
    """
    model, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer)
    intervention_applier = ModelInterventionApplier(model)
    suite = InterventionSuite(model, tokenizer, intervention_applier, formatter)
    
    # A dummy direction vector is needed to call the method
    dummy_vector = t.randn(model.config.hidden_size).to(model.device)
    dummy_direction = DirectionVector(vector=dummy_vector, layer=0, position_index=-1, score=0.0)

    # 1. Get baseline output from the suite
    baseline_results = suite.test_generation(
        direction=dummy_direction,
        test_prompts=prompts,
        intervention_type="ablate", # This type is arbitrary
        strengths=[0.0], # A strength of 0.0 should be a no-op
        max_new_tokens=10
    )
    
    baseline_texts = [res['generated_text'] for res in baseline_results['baseline_no_intervention']]
    strength_zero_texts = [res['generated_text'] for res in baseline_results['ablate_strength_0.0']]

    # 2. Assertions
    assert isinstance(baseline_texts, list)
    assert len(baseline_texts) == len(prompts)
    assert baseline_texts == strength_zero_texts, \
        "A zero-strength intervention should not change the generated output."
    

# --- Basic competence testing ---

# A small, diverse set of factual prompts to test basic knowledge.
# Each tuple is (prompt, expected_completion)
FACTUAL_PROMPTS = [
    ("The capital of Italy is", "Rome"),
    ("The capital of France is", "Paris"),
    ("The chemical symbol for water is", "H2O"),
    ("The first person to walk on the moon was Neil", "Armstrong"),
    ("To be or not to be, that is the", "question"),
    ("The main component of Earth's atmosphere is", "nitrogen"),
    ("The author of 'Pride and Prejudice' is Jane", "Austen"),
]

class FactualRecallWarning(UserWarning):
    """Custom warning for when a model fails too many factual recall tests."""
    pass

def test_factual_recall(model_and_tokenizer):
    """
    Tests the model's ability to answer basic factual questions by checking if the
    expected answer is present in a short generation. Issues a warning if the model
    fails on more than one prompt, but does not fail the test.
    """
    model, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer)
    
    failures = []

    for prompt, expected_answer in FACTUAL_PROMPTS:
        batch = formatter.format_batch([prompt])
        input_ids = batch['input_ids'].to(model.device)
        
        # we generate a few tokens so that answers starting with 'the' are allowed.
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id 
        )
        
        # Decode the generated tokens
        decoded_completion = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        # We check for containment ('in') instead of equality ('startswith').
        # This is more robust to conversational outputs like " The capital is Rome."
        if expected_answer.lower() not in decoded_completion.lower():
            failures.append({
                "prompt": prompt,
                "expected": expected_answer,
                "actual": decoded_completion
            })

    # Warning and Logging Logic
    if len(failures) > 1:
        model_name = model.config._name_or_path
        
        failure_log = "\n" + "="*20 + " Factual Recall Failure Log " + "="*20
        failure_log += f"\nModel: {model_name}"
        failure_log += f"\nFailed {len(failures)} of {len(FACTUAL_PROMPTS)} factual recall tests."
        failure_log += "\n" + "-"*60
        for failure in failures:
            failure_log += f"\n  Prompt:   '{failure['prompt']}'"
            failure_log += f"\n  Expected: '{failure['expected']}'"
            failure_log += f"\n  Actual:   '{failure['actual']}'\n"
        failure_log += "="*62

        warnings.warn(failure_log, FactualRecallWarning)


DTYPES_TO_TEST = [
    # pytest.param(t.bfloat16, id="bfloat16"),
    # pytest.param(t.float32, id="float32"),
    pytest.param("auto", id="auto")
]
# A list of prompts with different lengths to force padding
PADDED_BATCH_PROMPTS = [
    "This is a short sentence.",
    "This is a significantly longer sentence that will require the tokenizer to apply padding to the shorter one in the batch.",
    "Why is the sky blue?"
]

def test_logit_stability(model_and_tokenizer):
    """
    Tests for NaN/Inf in logits, which can indicate hardware-specific precision issues.
    This test is designed to run everywhere, but we expect it to fail for specific
    models on specific hardware (like Gemma on MPS).
    """
    model,tokenizer = model_and_tokenizer

    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
        
    formatter = ChatPromptFormatter(tokenizer)
    
    for prompt in PADDED_BATCH_PROMPTS:
        batch = formatter.format_batch([prompt])
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # --- The core of the test ---
        with t.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # --- The assertion that will fail on MPS ---
        has_nan = t.isnan(logits).any()
        has_inf = t.isinf(logits).any()
        
        assert not (has_nan or has_inf), f"Logits contain NaN or Inf values on {device}."


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
def test_batched_logit_stability_with_padding(model_and_tokenizer, dtype):
    """
    A more rigorous test that mimics the full script's batching behavior.
    It checks for NaN/Inf in logits when processing a padded batch of prompts
    with varying lengths. This is a common failure point for MPS (i.e. Apple Silicon)
    """
    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
    
    model,tokenizer = model_and_tokenizer
        
    formatter = ChatPromptFormatter(tokenizer)
    batch = formatter.format_batch(PADDED_BATCH_PROMPTS)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with t.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    has_nan = t.isnan(logits).any()
    has_inf = t.isinf(logits).any()
    
    assert not (has_nan or has_inf), (
        f"PADDED BATCH produced NaN/Inf on {device} with dtype {str(dtype)}"
    )

    # --- Breakdown Test 1: Unit Test the Formatter ---
def test_formatter_output_structure(model_and_tokenizer):
    """
    Tests if the ChatPromptFormatter produces a batch with the correct structure.
    This test does NOT involve a model forward pass.
    """
    _, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer) # This will modify the tokenizer instance
    
    batch = formatter.format_batch(PADDED_BATCH_PROMPTS)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    assert input_ids.shape == attention_mask.shape
    assert input_ids.ndim == 2
    assert input_ids.shape[0] == len(PADDED_BATCH_PROMPTS)
    
    # FIX: New, correct assertions
    if formatter.prepend_bos:
        # For base models, the first token of EVERY prompt should now be the BOS token
        assert (input_ids[:, 0] == tokenizer.bos_token_id).all()
        # The first non-BOS token of the shortest prompt should be a pad token.
        assert input_ids[0, 1].item() == tokenizer.pad_token_id
        assert attention_mask[0, 1].item() == 0
    else:
        # For instruction-tuned models, the first token of the shortest prompt should be a pad token
        assert input_ids[0, 0].item() == tokenizer.pad_token_id
        assert attention_mask[0, 0].item() == 0

# --- Breakdown Test 2: Test Model with Manually Padded Input ---
@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
def test_model_with_manual_padding(model_and_tokenizer, dtype):
    """
    Tests if the model itself is stable when given a correctly padded batch,
    bypassing our ChatPromptFormatter.
    """
    model, tokenizer = model_and_tokenizer
    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"


    # Manually prepare the batch
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    inputs = tokenizer(PADDED_BATCH_PROMPTS, return_tensors="pt", padding=True).to(device)

    with t.no_grad():
        logits = model(**inputs).logits

    has_nan = t.isnan(logits).any()
    has_inf = t.isinf(logits).any()
    assert not (has_nan or has_inf), f"Model failed with MANUAL padding on {device} with dtype {dtype}"


# # --- Breakdown Test 3: The Original Failing Test (Integration) ---
# @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
# def test_model_with_formatter_padding(model_and_tokenizer, dtype):
#     """
#     This is the original failing test. It checks the integration of the
#     formatter and the model.
#     """
#     model,tokenizer = model_and_tokenizer

#     device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
    
#     formatter = ChatPromptFormatter(tokenizer)
#     batch = formatter.format_batch(PADDED_BATCH_PROMPTS)
#     inputs = {k: v.to(device) for k, v in batch.items()}

#     with t.no_grad():
#         logits = model(**inputs).logits
        
#     has_nan = t.isnan(logits).any()
#     has_inf = t.isinf(logits).any()
#     assert not (has_nan or has_inf), f"Model failed with FORMATTER padding on {device} with dtype {dtype}"
    
@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
def test_end_to_end_stability(model_and_tokenizer, dtype):
    """
    A robust integration test that verifies the end-to-end stability of the
    prompt formatting and model generation pipeline. This is the only test
    we need to check for hardware/padding-related NaN/Inf issues.
    """

    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
    
    model,tokenizer = model_and_tokenizer
        
    # 1. Use the REAL ChatPromptFormatter, the single source of truth
    formatter = ChatPromptFormatter(tokenizer)
    batch = formatter.format_batch(PADDED_BATCH_PROMPTS)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # 2. Run the model with the formatter's output
    with t.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
    # 3. Check for instability
    has_nan = t.isnan(logits).any()
    has_inf = t.isinf(logits).any()
    
    # 4. Assert with a highly informative failure message
    if has_nan or has_inf:
        # Construct a detailed error message for debugging
        model_short_name = model.name_or_path.split('/')[-1] if hasattr(model, 'name_or_path') else 'unknown_model'
        debug_message = (
            f"\n--- LOGIT STABILITY FAILURE ---"
            f"\nModel: {model_short_name}"
            f"\nDtype: {str(dtype)}"
            f"\nDevice: {device}"
            f"\nNaNs found: {has_nan.item()}"
            f"\nInfs found: {has_inf.item()}"
            f"\n\n--- Formatter Output ---"
            f"\nInput IDs shape: {input_ids.shape}"
            f"\nAttention Mask shape: {attention_mask.shape}"
            f"\nInput IDs (snippet):\n{input_ids[:, :10]}"
            f"\nAttention Mask (snippet):\n{attention_mask[:, :10]}"
            f"\n--------------------------"
        )
        pytest.fail(debug_message)

def run_stability_check(model, tokenizer, prompts):
    """Helper function to run a forward pass and check for NaNs/Infs."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with t.no_grad():
        logits = model(**inputs).logits
    return not (t.isnan(logits).any() or t.isinf(logits).any())


def test_stability_with_different_pad_tokens(model_and_tokenizer):
    """
    This test definitively isolates the pad_token issue.
    - It is expected to FAIL when pad_token is set to eos_token for unstable models.
    - It is expected to PASS when pad_token is set to a safe default (like unk_token).
    """
    # device unused?
    # device = "cuda" if t.cuda.is_available() else "cpu". 
    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"

    model, tokenizer = model_and_tokenizer
    tokenizer.padding_side = 'left'

    model_shortname = model.name_or_path.split('/')[-1]
    if tokenizer.pad_token is not None:
        s = f"Model {model_shortname} failed with built-in padding token."
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        s = f"Model {model_shortname} failed when unk_token used as padding"
    else:
        # If no unk_token, we must use eos_token and acknowledge the risk for some models
        tokenizer.pad_token = tokenizer.eos_token
        s = f"Model {model_shortname} failed when eos_token used as padding"

    is_stable_with_padding = run_stability_check(model, tokenizer, PADDED_BATCH_PROMPTS)
    assert is_stable_with_padding, s

# conclusion. Base models work. Chat models don't. eos tokens being used as padding are a red herring.
# Maybe consider applying no template to the chat models for a bit, acknowleging that results will be borked, but at least have them run?
def test_end_to_end_stability_with_formatter(model_and_tokenizer):
    """
    This is the final, robust integration test. It verifies the end-to-end
    stability of the application's true logic by using the ChatPromptFormatter
    to prepare the batch. It should now pass for all models.
    """
    model, tokenizer = model_and_tokenizer
    device = model.device

    model_shortname = model.name_or_path.split('/')[-1]

    # 1. Use the REAL ChatPromptFormatter, the single source of truth for your application.
    formatter = ChatPromptFormatter(tokenizer)
    
    # 2. Create the batch using the exact same logic as your main script.
    batch = formatter.format_batch(PADDED_BATCH_PROMPTS)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # 3. Run the model with the formatter's output.
    with t.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
    # 4. Assert that the output is stable.
    has_nan = t.isnan(logits).any()
    has_inf = t.isinf(logits).any()
    
    assert not (has_nan or has_inf), (
        f"Model '{model_shortname}' produced NaN/Inf logits with the ChatPromptFormatter. "
    )

# def test_stability_with_incorrect_pad_token_setup(model_and_tokenizer):
#     """
#     This test REPRODUCES the error. It manually creates the flawed condition.
#     We expect this test to FAIL, proving our diagnosis is correct.
#     """
#     model, tokenizer = model_and_tokenizer
    
#     # Manually create the flawed state: unset the pad_token
#     tokenizer.pad_token = None
    
#     # Now, tokenize. The Hugging Face tokenizer will likely default to an unsafe pad_token_id.
#     # We are not using our formatter here, just the raw tokenizer.
#     inputs = tokenizer(
#         PADDED_BATCH_PROMPTS, return_tensors="pt", padding=True, add_special_tokens=False
#     ).to(model.device)

#     with t.no_grad():
#         logits = model(**inputs).logits
        
#     has_nan_or_inf = t.isnan(logits).any() or t.isinf(logits).any()

#     assert not has_nan_or_inf, "Model failed to produce stable logits on old setup."
    
#     # # We use pytest.xfail to mark that we EXPECT this test to fail.
#     # # If it passes, pytest will report it as an "unexpected pass" (XPASS).
#     # if has_nan_or_inf:
#     #     pytest.xfail("Correctly reproduced NaN/Inf failure with incorrect pad token setup.")
    
#     # # If it somehow doesn't fail, we raise an error.
#     # assert not has_nan_or_inf, "Model was expected to fail but produced stable logits."

# def test_stability_with_correct_pad_token_setup(model_and_tokenizer):
#     """
#     This test FIXES the error. It ensures the pad token is set correctly
#     before tokenization, just like our improved ChatPromptFormatter does.
#     We expect this test to PASS.
#     """
#     model, tokenizer = model_and_tokenizer
    
#     # Manually create the CORRECT state
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'left'
    
#     inputs = tokenizer(
#         PADDED_BATCH_PROMPTS, return_tensors="pt", padding=True, add_special_tokens=False
#     ).to(model.device)

#     with t.no_grad():
#         logits = model(**inputs).logits
        
#     has_nan_or_inf = t.isnan(logits).any() or t.isinf(logits).any()
    
#     assert not has_nan_or_inf, "Model failed to produce stable logits even with correct pad token setup."