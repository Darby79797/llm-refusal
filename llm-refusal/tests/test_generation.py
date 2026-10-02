import pytest
import torch as t
from generation import generate_with_hooks
from formatting import ChatPromptFormatter
from interventions import ModelInterventionApplier
from evaluation import InterventionSuite
from datatypes import DirectionVector

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

    # Check that padding was applied correctly (RIGHT padding — real tokens start
    # at index 0 in every row, so true_len-based indexing is valid everywhere)
    assert input_ids.shape[1] > tokenizer(prompts[0], return_tensors='pt')['input_ids'].shape[1]

    assert attention_mask[0, 0].item() == 1, "First token of shorter prompt should be attended to."
    assert attention_mask[1, 0].item() == 1, "First token of longer prompt should be attended to."
    if formatter.prepend_bos:
        assert input_ids[0, 0].item() == tokenizer.bos_token_id, "First token of shorter prompt should be BOS."
        assert input_ids[1, 0].item() == tokenizer.bos_token_id, "First token of longer prompt should be BOS."

    # The shorter row is padded at the END; the longer row is not padded at all
    assert attention_mask[0, -1].item() == 0, "The last token of the shorter prompt should be masked."
    assert attention_mask[1, -1].item() == 1, "The last token of the longer prompt should not be masked."

    # Positions must be derived from the mask, not from the padded width
    assert batch['position_ids'][1].tolist() == list(range(input_ids.shape[1]))
    true_len_0 = int(attention_mask[0].sum())
    assert batch['position_ids'][0, :true_len_0].tolist() == list(range(true_len_0))

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

def test_batched_generation_matches_individual(model_and_tokenizer):
    """Generation must be batch-invariant.

    We right-pad, so model.generate() would read the next token from a pad
    position for every row shorter than the longest in the batch — those rows
    generate from a pad token and drift off-task. Measured on Qwen2.5-3B, this
    moved the baseline refusal rate from 95% to 70%. Evaluation therefore routes
    through generate_with_hooks(), and this test pins that property down.
    """
    model, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer)

    prompts = [
        "Name a color.",
        "Explain, in a couple of sentences, why the sky appears blue during the day.",
    ]

    individual = [
        generate_with_hooks(model, tokenizer, formatter, [p], max_new_tokens=20)[0]
        for p in prompts
    ]
    batched = generate_with_hooks(model, tokenizer, formatter, prompts, max_new_tokens=20)

    assert batched == individual, (
        "Batched generation diverged from individual generation — the padded row "
        "is being decoded from a pad position."
    )


def test_generation_ignores_shipped_sampling_config(model_and_tokenizer):
    """Decoding must be plain greedy, as the paper specifies.

    model.generate() applies the model's shipped generation_config, and Qwen2.5
    ships repetition_penalty 1.05-1.1 — a logits processor, so it applies even
    with do_sample=False. generate_with_hooks() does pure argmax, which must
    equal model.generate() only once that penalty is explicitly disabled.
    """
    model, tokenizer = model_and_tokenizer
    formatter = ChatPromptFormatter(tokenizer)
    prompt = "Explain what photosynthesis is."

    batch = formatter.format_batch([prompt])
    ids = batch['input_ids'].to(model.device)
    mask = batch['attention_mask'].to(model.device)
    with t.no_grad():
        out = model.generate(
            ids, attention_mask=mask, max_new_tokens=16, do_sample=False,
            repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    hf_text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    hook_text = generate_with_hooks(model, tokenizer, formatter, [prompt], max_new_tokens=16)[0]

    assert hook_text == hf_text or hf_text.startswith(hook_text), (
        f"pure-greedy decode diverged from model.generate(repetition_penalty=1.0):\n"
        f"  hf   : {hf_text!r}\n  hook : {hook_text!r}"
    )


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
    

# A list of prompts with different lengths to force padding
PADDED_BATCH_PROMPTS = [
    "This is a short sentence.",
    "This is a significantly longer sentence that will require the tokenizer to apply padding to the shorter one in the batch.",
    "Why is the sky blue?"
]

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
