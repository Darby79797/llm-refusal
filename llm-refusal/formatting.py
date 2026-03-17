import torch as t
from typing import List, Dict
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


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
        self.tokenizer.padding_side = 'left'

        self.is_instruction_tuned = any(tag in tokenizer.name_or_path.lower() for tag in ["-it", "-instruct", "-chat"])

        max_len = self.tokenizer.model_max_length
        if max_len > 100000:
            logger.warning(f"Tokenizer's model_max_length is a large sentinel value ({max_len}). Setting a safe default of 4096.")
            self.safe_max_length = 4096
        else:
            self.safe_max_length = max_len

        # --- CRITICAL: Explicitly define if a BOS token is needed ---
        # Base models like GPT-2 benefit from an explicit BOS token.
        self.prepend_bos = not self.is_instruction_tuned

        # Determine the template
        if self.is_instruction_tuned:
            if tokenizer.chat_template:
                self.template = None # Signal to use built-in template
            else:
                # Manual templates for known instruction-tuned models
                model_name = tokenizer.name_or_path.lower()
                if "gemma" in model_name:
                    self.template = "<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n"
                elif "qwen1.5" in model_name:
                    self.template = "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
                elif "qwen" in model_name:
                    self.template = "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
                elif "yi" in model_name:
                    self.template = "<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n"
                elif "llama-3" in model_name:
                    self.template = "<|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                elif "llama-2" in model_name:
                    self.template = "[INST] {x} [/INST]" # Note the space before [/INST]
        else:
            # Base models get a pass-through template
            self.template = "{x}"

        # --- Compute assistant prefix token count ---
        # Number of tokens between the end of the user's instruction and the
        # final token of the formatted prompt (i.e. the assistant turn suffix).
        # Used by search to auto-derive max_positions covering -1 through the
        # end-of-instruction (EOI) token position.
        self.assistant_prefix_tokens = self._compute_assistant_prefix_tokens()

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

    def format_batch(self, prompts: List[str]) -> Dict[str, t.Tensor]:
        """
        Formats a batch of prompts, applying the chat template then tokenizing.
        """
        # --- Apply chat template ---
        if self.template is not None:
            # Manual template (instruction-tuned) or pass-through ("{x}" for base models)
            formatted_prompts = [self.template.format(x=p) for p in prompts]
        else:
            # Use tokenizer's built-in chat_template
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': p}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for p in prompts
            ]

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

        return {'input_ids': input_ids, 'attention_mask': attention_mask}
