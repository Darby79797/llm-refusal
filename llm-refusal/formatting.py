import torch as t
from typing import List, Dict
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# ── Padding invariant ────────────────────────────────────────────────────────
# The whole pipeline RIGHT-pads: real tokens occupy indices 0..true_len-1 and
# pads follow. Every consumer therefore locates the generation boundary as
# `attention_mask.sum(-1) - 1`, and activations at index `true_len + pos_idx`.
#
# That arithmetic is silently WRONG under left padding — it reads a token from
# the middle of the prompt — and this repo has shipped both failure modes:
#   - left-padded batches indexed as if right-padded (the April 2026 scoring bug)
#   - right-padded batches fed to model.generate(), which reads the LAST column
#     and so decodes from a pad slot (the August 2026 evaluation bug)
# Neither raised; both silently produced plausible numbers. So the invariant is
# asserted here rather than assumed, and every consumer goes through these
# helpers. See RESULTS.md "Two Padding Bugs".


def assert_right_padded(attention_mask: t.Tensor) -> None:
    """Raise unless every row is a contiguous run of 1s followed by 0s.

    A left-padded (or interior-masked) batch fails here instead of silently
    producing off-by-true_len indexing downstream.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be 2D (batch, seq), got shape {tuple(attention_mask.shape)}")
    # Right-padded <=> mask is non-increasing along the sequence dimension.
    if attention_mask.shape[1] > 1 and not bool((attention_mask[:, :-1] >= attention_mask[:, 1:]).all()):
        bad = (~(attention_mask[:, :-1] >= attention_mask[:, 1:]).all(dim=1)).nonzero().flatten().tolist()
        raise ValueError(
            f"attention_mask is not right-padded (rows {bad[:5]} have a 0 before a 1). "
            "This pipeline right-pads; left-padded batches break last-real-token indexing "
            "in scoring, activations and generation. See RESULTS.md 'Two Padding Bugs'."
        )
    if not bool((attention_mask.sum(dim=1) > 0).all()):
        raise ValueError("attention_mask has a row with no real tokens.")


def last_real_token_indices(attention_mask: t.Tensor) -> t.Tensor:
    """Index of the final non-pad token in each row, validating right-padding.

    Use this everywhere the generation boundary is read. Never hand-roll
    `attention_mask.sum(-1) - 1`: that expression is correct only under the
    invariant this function checks.
    """
    assert_right_padded(attention_mask)
    return attention_mask.sum(dim=1) - 1


class ChatPromptFormatter:
    """
    A helper class to correctly format prompts.
    It now explicitly handles BOS tokens for greater predictability.
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        # --- Set padding token and side ---
        if self.tokenizer.pad_token is None:
            logger.info("Tokenizer has no pad_token. Setting to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        self.is_instruction_tuned = any(tag in tokenizer.name_or_path.lower() for tag in ["-it", "-instruct", "-chat"])

        max_len = self.tokenizer.model_max_length
        if max_len > 100000:
            logger.warning(f"Tokenizer's model_max_length is a large sentinel value ({max_len}). Setting a safe default of 4096.")
            self.safe_max_length = 4096
        else:
            self.safe_max_length = max_len

        # Determine the template
        # Check for known models FIRST (manual templates override built-in ones
        # when the built-in template has known issues, e.g. Llama-2's doesn't
        # respond to add_generation_prompt).
        model_name = tokenizer.name_or_path.lower()
        if self.is_instruction_tuned:
            manual_template = self._get_manual_template(model_name, bool(tokenizer.chat_template))
            if manual_template is not None:
                self.template = manual_template
                # Manual templates don't include BOS — prepend it
                self.prepend_bos = True
            elif tokenizer.chat_template:
                self.template = None # Signal to use built-in template
                self.prepend_bos = False  # Built-in templates include BOS
            else:
                self.template = None
                self.prepend_bos = False
                logger.warning(f"No chat template found for {tokenizer.name_or_path}. "
                               "Generation may be incorrect.")
        else:
            # Base models get a pass-through template with explicit BOS
            self.template = "{x}"
            self.prepend_bos = True

        # --- Compute assistant prefix token count ---
        # Number of tokens between the end of the user's instruction and the
        # final token of the formatted prompt (i.e. the assistant turn suffix).
        # Used by search to auto-derive max_positions covering -1 through the
        # end-of-instruction (EOI) token position.
        self.assistant_prefix_tokens = self._compute_assistant_prefix_tokens()

    @staticmethod
    def _get_manual_template(model_name: str, has_builtin: bool):
        """Return a manual chat template for known models, or None to use built-in.

        Only overrides the built-in template when it has known issues
        (e.g. Llama-2's doesn't respond to add_generation_prompt).
        For models without a built-in template, provides a fallback.
        """
        # Models with broken built-in templates — always override
        if "llama-2" in model_name:
            # Trailing space matches Arditi's format — the model's assistant turn
            # starts with a space, so including it in the prompt means pos -1 is
            # the actual generation boundary where the refusal decision is made.
            return "[INST] {x} [/INST] "

        # For everything below, only use manual templates as fallbacks
        # when no built-in template exists
        if has_builtin:
            return None

        if "gemma" in model_name:
            return "<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n"
        elif "qwen1.5" in model_name:
            return "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
        elif "qwen" in model_name:
            return "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
        elif "yi" in model_name:
            return "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama-3" in model_name:
            return "<|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return None

    def _compute_assistant_prefix_tokens(self) -> int:
        """Count tokens after the user instruction in the formatted prompt.

        Uses a dummy instruction to measure how many tokens the chat template
        appends after the instruction content. For base models (template="{x}")
        this returns 0.
        """
        dummy = "DUMMY_INSTRUCTION_MARKER"

        if self.template is not None:
            # Manual template or pass-through ("{x}")
            formatted = self.template.format(x=dummy)
        else:
            # Built-in chat_template
            formatted = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': dummy}],
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize the full formatted prompt and the instruction alone
        full_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        instruction_ids = self.tokenizer.encode(dummy, add_special_tokens=False)

        # Find where the instruction tokens end in the full sequence
        # Search for the instruction token subsequence
        instr_len = len(instruction_ids)
        match_pos = None
        for i in range(len(full_ids) - instr_len + 1):
            if full_ids[i:i + instr_len] == instruction_ids:
                match_pos = i
                break

        if match_pos is not None:
            suffix_tokens = len(full_ids) - (match_pos + instr_len)
        else:
            # Fallback: tokenize just the suffix portion of the template
            if self.template is not None and "{x}" in self.template:
                suffix_str = self.template.split("{x}")[-1]
                suffix_tokens = len(self.tokenizer.encode(suffix_str, add_special_tokens=False))
            else:
                suffix_tokens = 0
                logger.warning("Could not determine assistant prefix tokens; defaulting to 0.")

        logger.info(f"Assistant prefix tokens: {suffix_tokens} (tokens between EOI and position -1)")
        return suffix_tokens

    def format_text(self, prompt: str) -> str:
        """The templated prompt string, ending at the generation boundary (no BOS:
        callers that tokenize it themselves must prepend BOS when `prepend_bos`)."""
        if self.template is not None:
            # Manual template (instruction-tuned) or pass-through ("{x}" for base models)
            return self.template.format(x=prompt)
        # Tokenizer's built-in chat_template
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)

    def format_batch(self, prompts: List[str]) -> Dict[str, t.Tensor]:
        """
        Formats a batch of prompts, applying the chat template then tokenizing.
        """
        # Re-assert on every call, not just in __init__: the tokenizer is a shared
        # mutable object, and other consumers flip this (lm_eval's HFLM sets its
        # own padding side for generation tasks). If it were flipped between
        # __init__ and here, every downstream last-real-token index would be wrong
        # while everything still ran.
        if self.tokenizer.padding_side != 'right':
            logger.warning(f"tokenizer.padding_side was changed to "
                           f"{self.tokenizer.padding_side!r}; restoring 'right'.")
            self.tokenizer.padding_side = 'right'

        formatted_prompts = [self.format_text(p) for p in prompts]

        # --- Tokenize (special tokens disabled — we manage them ourselves) ---
        tokenized_output = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.safe_max_length,
            add_special_tokens=False
        )

        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']

        # --- Manually add BOS token if required ---
        if self.prepend_bos:
            bos_tensor = t.full((input_ids.shape[0], 1), self.tokenizer.bos_token_id, dtype=t.long)
            input_ids = t.cat([bos_tensor, input_ids], dim=1)

            mask_tensor = t.ones((attention_mask.shape[0], 1), dtype=t.long)
            attention_mask = t.cat([mask_tensor, attention_mask], dim=1)

            # Ensure we don't exceed max length after adding BOS
            if input_ids.shape[1] > self.tokenizer.model_max_length:
                input_ids = input_ids[:, -self.tokenizer.model_max_length:]
                attention_mask = attention_mask[:, -self.tokenizer.model_max_length:]

        # --- Compute position_ids explicitly (we right-pad, not left-pad) ---
        # HuggingFace does not derive correct RoPE position_ids from
        # attention_mask on its own for a batched forward pass — without
        # explicit position_ids, every row is assumed to start at position 0,
        # which silently corrupts the model's RoPE encodings for any batch
        # containing padding. We right-pad (real tokens always start at
        # position 0) and derive positions as cumsum(attention_mask) - 1;
        # padded slots get an inert sentinel position (1) since they're
        # masked out of attention and their outputs are discarded anyway.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
