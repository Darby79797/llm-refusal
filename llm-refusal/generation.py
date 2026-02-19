import torch as t
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from formatting import ChatPromptFormatter


def generate_with_hooks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_formatter: ChatPromptFormatter, # Now passed in as an argument
    prompts: List[str], # Now takes raw prompts
    max_new_tokens: int = 64
) -> List[str]:
    """
    A robust, manual greedy decoding loop that correctly handles KV caching
    and works reliably with hooks. This replaces the in-built .generate() method.
    """
    batch = prompt_formatter.format_batch(prompts)
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)

    batch_size = input_ids.shape[0]
    generated_ids_list = [[] for _ in range(batch_size)]
    finished_sequences = [False] * batch_size

    with t.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = t.argmax(next_token_logits, dim=-1)

        for _ in range(max_new_tokens):
            if all(finished_sequences): break

            for i in range(batch_size):
                if not finished_sequences[i]:
                    token_id = next_token_ids[i].item()
                    if token_id == tokenizer.eos_token_id: finished_sequences[i] = True
                    else: generated_ids_list[i].append(token_id)

            if all(finished_sequences): break

            current_input_ids = next_token_ids.unsqueeze(-1)
            attention_mask = t.cat([attention_mask, t.ones(batch_size, 1, device=model.device)], dim=1)

            outputs = model(input_ids=current_input_ids, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = t.argmax(next_token_logits, dim=-1)

    return tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True)
