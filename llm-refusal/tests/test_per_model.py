"""
Per-model correctness tests.

These tests verify invariants that must hold for every model we experiment on.
They catch bugs like:
  - Left-padding logit corruption (the position_ids / RoPE bug)
  - Chat template not producing a generation prompt (the Llama-2 bug)
  - Activation extraction returning wrong values for padded sequences

Run with:
  pytest llm-refusal/tests/test_per_model.py -v
  pytest llm-refusal/tests/test_per_model.py -v -k "Llama-3"
"""
import pytest
import torch as t

from transformers import AutoModelForCausalLM, AutoTokenizer
from formatting import ChatPromptFormatter

# ── Models to test ──────────────────────────────────────────────────────────
# Each model we've experimented on or plan to experiment on.
# Tests are parametrized over this list — add new models here.
PER_MODEL_TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
]


@pytest.fixture(scope="session", params=PER_MODEL_TEST_MODELS)
def per_model_setup(request):
    """Load model, tokenizer, and formatter once per model per session."""
    model_name = request.param
    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=t.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load {model_name}: {e}")
    formatter = ChatPromptFormatter(tokenizer)
    return model, tokenizer, formatter, model_name


# ── 1. Padded batch logit consistency ────────────────────────────────────────
# The bug: with incorrect padding handling, shorter prompts in a batch get
# garbage logits while the longest prompt (no padding) is fine.

class TestPaddedBatchConsistency:
    """Batched logits at the last real token must match individual processing."""

    SHORT_PROMPT = "What is 2+2?"
    LONG_PROMPT = (
        "Explain in detail how photosynthesis works in plants, "
        "including the light-dependent and light-independent reactions, "
        "the role of chlorophyll, and the overall chemical equation."
    )

    def _get_top_tokens(self, model, tokenizer, formatter, prompt, batch=None):
        """Get top-5 token IDs for a prompt, either individually or from a batch."""
        if batch is not None:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            position_ids = batch['position_ids'].to(model.device)
            with t.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            last = attention_mask.sum(dim=1) - 1
            # Find which index this prompt is at
            idx = None
            for i in range(input_ids.shape[0]):
                # Match by sequence length
                if batch['_prompts'][i] == prompt:
                    idx = i
                    break
            assert idx is not None
            logits = out.logits[idx, last[idx], :]
        else:
            batch_single = formatter.format_batch([prompt])
            input_ids = batch_single['input_ids'].to(model.device)
            attention_mask = batch_single['attention_mask'].to(model.device)
            position_ids = batch_single['position_ids'].to(model.device)
            with t.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            last = attention_mask.sum(dim=1) - 1
            logits = out.logits[0, last[0], :]
        return t.topk(logits.float(), 5).indices.tolist()

    def test_short_prompt_logits_match_when_batched(self, per_model_setup):
        """The SHORT prompt must produce the same top tokens whether processed
        alone or batched with a longer prompt (which forces padding)."""
        model, tokenizer, formatter, model_name = per_model_setup

        # Individual
        individual_top5 = self._get_top_tokens(model, tokenizer, formatter, self.SHORT_PROMPT)

        # Batched (short will get padding)
        batch = formatter.format_batch([self.SHORT_PROMPT, self.LONG_PROMPT])
        batch['_prompts'] = [self.SHORT_PROMPT, self.LONG_PROMPT]
        batched_top5 = self._get_top_tokens(model, tokenizer, formatter, self.SHORT_PROMPT, batch=batch)

        # Top-1 must match; top-5 should mostly overlap
        assert individual_top5[0] == batched_top5[0], (
            f"[{model_name}] Top-1 token differs: "
            f"individual={tokenizer.decode([individual_top5[0]])} "
            f"vs batched={tokenizer.decode([batched_top5[0]])}"
        )
        overlap = len(set(individual_top5) & set(batched_top5))
        assert overlap >= 3, (
            f"[{model_name}] Only {overlap}/5 top tokens overlap between "
            f"individual and batched processing. "
            f"Individual: {[tokenizer.decode([i]) for i in individual_top5]}, "
            f"Batched: {[tokenizer.decode([i]) for i in batched_top5]}"
        )


# ── 1b. Left-padding regression test ────────────────────────────────────────
# This test checks which models produce WRONG logits under left-padding
# (the old default). Models that fail here had corrupted results in all
# prior experiments. We expect Llama-3 and Llama-3.1 to fail.

class TestLeftPaddingRegression:
    """Identifies models whose prior results were corrupted by left-padding."""

    SHORT_PROMPT = "What is 2+2?"
    LONG_PROMPT = (
        "Explain in detail how photosynthesis works in plants, "
        "including the light-dependent and light-independent reactions, "
        "the role of chlorophyll, and the overall chemical equation."
    )

    def test_left_padding_corrupts_logits(self, per_model_setup):
        """Check if left-padding produces different top-1 token vs individual.
        Models that FAIL this test had corrupted prior results and need re-running.
        Models that PASS were unaffected by the left-padding bug."""
        model, tokenizer, formatter, model_name = per_model_setup

        # Get individual (no padding) top-5 for the short prompt
        # Use the formatter for template, but process individually
        batch_single = formatter.format_batch([self.SHORT_PROMPT])
        ids_single = batch_single['input_ids'].to(model.device)
        mask_single = batch_single['attention_mask'].to(model.device)
        pos_single = batch_single['position_ids'].to(model.device)
        with t.no_grad():
            out_single = model(input_ids=ids_single, attention_mask=mask_single, position_ids=pos_single)
        last_single = mask_single.sum(dim=1) - 1
        individual_top1 = t.argmax(out_single.logits[0, last_single[0], :].float()).item()

        # Now manually left-pad a batch (simulating old behavior: no position_ids)
        short_formatted = formatter.format_batch([self.SHORT_PROMPT])
        long_formatted = formatter.format_batch([self.LONG_PROMPT])
        short_ids = short_formatted['input_ids'][0]
        long_ids = long_formatted['input_ids'][0]

        # Left-pad the short one to match long's length
        pad_len = long_ids.shape[0] - short_ids.shape[0]
        if pad_len > 0:
            pad_ids = t.full((pad_len,), tokenizer.pad_token_id, dtype=t.long)
            padded_short_ids = t.cat([pad_ids, short_ids])
            padded_short_mask = t.cat([t.zeros(pad_len, dtype=t.long), t.ones_like(short_ids)])
        else:
            padded_short_ids = short_ids
            padded_short_mask = t.ones_like(short_ids)

        batch_ids = t.stack([padded_short_ids, long_ids]).to(model.device)
        batch_mask = t.stack([padded_short_mask, long_formatted['attention_mask'][0]]).to(model.device)

        # NO position_ids — this is the old broken behavior
        with t.no_grad():
            out_batch = model(input_ids=batch_ids, attention_mask=batch_mask)
        last_batch = batch_mask.sum(dim=1) - 1
        batched_top1 = t.argmax(out_batch.logits[0, last_batch[0], :].float()).item()

        if individual_top1 != batched_top1:
            pytest.xfail(
                f"[{model_name}] LEFT-PADDING BUG CONFIRMED: "
                f"individual top-1={repr(tokenizer.decode([individual_top1]))} "
                f"vs left-padded={repr(tokenizer.decode([batched_top1]))}. "
                f"ALL PRIOR RESULTS FOR THIS MODEL WERE CORRUPTED."
            )


# ── 2. Chat template produces generation boundary ───────────────────────────
# The bug: Llama-2's HuggingFace template doesn't respond to
# add_generation_prompt, so the formatted prompt ends with "]" instead of
# a space token at the generation boundary.

class TestChatTemplateGenerationBoundary:
    """The formatted prompt must end with the assistant turn prefix,
    not with instruction content or closing brackets."""

    def test_template_ends_with_assistant_header(self, per_model_setup):
        """After formatting, the last tokens should be the assistant header
        (not user content or closing brackets like ']')."""
        model, tokenizer, formatter, model_name = per_model_setup

        if not formatter.is_instruction_tuned:
            pytest.skip("Base model — no chat template")

        prompt = "Tell me a joke"
        batch = formatter.format_batch([prompt])
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]

        # Get last real token
        last_idx = attention_mask.sum() - 1
        last_token_id = input_ids[last_idx].item()
        last_token = tokenizer.decode([last_token_id])

        # The last token should NOT be part of the user instruction
        assert "joke" not in last_token.lower(), (
            f"[{model_name}] Last token is '{last_token}' — instruction content leaked "
            f"into the generation boundary position. Chat template may not be adding "
            f"a generation prompt."
        )

        # The last token should NOT be a closing bracket
        assert last_token.strip() not in ["]", "}", ">", ">>"], (
            f"[{model_name}] Last token is '{last_token}' — looks like the chat template "
            f"doesn't add a proper generation prompt (similar to Llama-2 bug)."
        )

    def test_assistant_prefix_tokens_positive(self, per_model_setup):
        """Instruction-tuned models must have >0 assistant prefix tokens,
        meaning the template adds tokens after the instruction."""
        _, _, formatter, model_name = per_model_setup

        if not formatter.is_instruction_tuned:
            pytest.skip("Base model — no chat template")

        assert formatter.assistant_prefix_tokens > 0, (
            f"[{model_name}] assistant_prefix_tokens is 0 — the chat template adds "
            f"nothing after the instruction. This means pos=-1 is the last instruction "
            f"token, not the generation boundary."
        )

    def test_different_prompts_same_suffix(self, per_model_setup):
        """The template suffix (after instruction) should be identical regardless
        of instruction content. This catches templates that vary per-prompt."""
        _, tokenizer, formatter, model_name = per_model_setup

        if not formatter.is_instruction_tuned:
            pytest.skip("Base model — no chat template")

        prompts = ["Hello", "Explain quantum mechanics in detail"]
        batches = [formatter.format_batch([p]) for p in prompts]

        suffix_len = formatter.assistant_prefix_tokens
        for i, (batch, prompt) in enumerate(zip(batches, prompts)):
            ids = batch['input_ids'][0]
            mask = batch['attention_mask'][0]
            real_len = mask.sum().item()
            suffix = ids[real_len - suffix_len: real_len].tolist()
            if i == 0:
                reference_suffix = suffix
            else:
                assert suffix == reference_suffix, (
                    f"[{model_name}] Template suffix differs between prompts: "
                    f"'{prompts[0]}' -> {reference_suffix}, "
                    f"'{prompt}' -> {suffix}"
                )


# ── 3. Position IDs correctness ─────────────────────────────────────────────
# Verifies that format_batch produces correct position_ids that account
# for padding, and that these are actually used by the model.

class TestPositionIds:
    """format_batch must produce correct position_ids for padded batches."""

    def test_position_ids_present(self, per_model_setup):
        """format_batch must return position_ids in the output dict."""
        _, _, formatter, _ = per_model_setup
        batch = formatter.format_batch(["Hello"])
        assert 'position_ids' in batch, "format_batch must return position_ids"

    def test_position_ids_match_attention_mask(self, per_model_setup):
        """For right-padding: real tokens get positions 0..N-1,
        padding tokens get position 1 (the masked-fill value)."""
        _, _, formatter, model_name = per_model_setup

        batch = formatter.format_batch(["Short", "This is a much longer prompt to test padding"])
        pos = batch['position_ids']
        mask = batch['attention_mask']

        for i in range(2):
            real_len = mask[i].sum().item()
            real_positions = pos[i, :real_len].tolist()

            # Real tokens should have sequential positions starting from 0
            assert real_positions == list(range(real_len)), (
                f"[{model_name}] Row {i}: real token positions should be 0..{real_len-1}, "
                f"got {real_positions[:10]}..."
            )

            # Padding positions should all be 1 (masked fill value)
            if real_len < pos.shape[1]:
                pad_positions = pos[i, real_len:].tolist()
                assert all(p == 1 for p in pad_positions), (
                    f"[{model_name}] Row {i}: padding positions should all be 1, "
                    f"got {pad_positions[:5]}..."
                )

    def test_unpadded_batch_has_simple_range(self, per_model_setup):
        """A single prompt (no padding) should have position_ids = 0..N-1."""
        _, _, formatter, model_name = per_model_setup
        batch = formatter.format_batch(["Hello world"])
        pos = batch['position_ids'][0]
        expected = list(range(pos.shape[0]))
        assert pos.tolist() == expected, (
            f"[{model_name}] Single prompt position_ids should be simple range"
        )


# ── 4. Activation extraction position consistency ───────────────────────────
# The extracted activation at a given (layer, position) must be the same
# whether the prompt is processed alone or in a padded batch.

class TestActivationExtractionConsistency:
    """Activations at the last real token must match between individual
    and batched processing."""

    def test_last_token_activation_matches(self, per_model_setup):
        """Activation at pos=-1 (last real token) must be identical whether
        the prompt is processed individually or in a padded batch."""
        model, tokenizer, formatter, model_name = per_model_setup

        short = "What is gravity?"
        long = "Explain the theory of general relativity including spacetime curvature and geodesics"

        def get_last_activation(prompts, target_idx=0):
            batch = formatter.format_batch(prompts)
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            position_ids = batch['position_ids'].to(model.device)

            captured = {}
            hook = model.model.layers[0].register_forward_pre_hook(
                lambda mod, args: captured.update({'act': args[0].clone()})
            )
            with t.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            hook.remove()

            last_idx = attention_mask[target_idx].sum().item() - 1
            return captured['act'][target_idx, last_idx, :].float().cpu()

        # Get activation for short prompt: individually vs batched
        act_individual = get_last_activation([short], target_idx=0)
        act_batched = get_last_activation([short, long], target_idx=0)

        cos_sim = t.nn.functional.cosine_similarity(
            act_individual.unsqueeze(0), act_batched.unsqueeze(0)
        ).item()

        assert cos_sim > 0.99, (
            f"[{model_name}] Activation at last token diverges between individual "
            f"and batched processing (cosine similarity = {cos_sim:.4f}). "
            f"This suggests padding is corrupting the representations."
        )
