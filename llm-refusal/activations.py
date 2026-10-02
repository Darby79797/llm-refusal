import torch as t
from typing import List, Dict, Tuple
import logging

from formatting import ChatPromptFormatter, assert_right_padded
from prompts import COMPLETION_SEP

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extracts residual stream activations from a model, using batching.
    Activations are captured by forward pre-hooks, i.e. the residual stream entering each layer."""
    def __init__(self, model, tokenizer, transformer_layers, prompt_formatter: ChatPromptFormatter):
        self.model = model
        self.tokenizer = tokenizer
        self.transformer_layers = transformer_layers
        self.prompt_formatter = prompt_formatter
        self.device = self.model.device

    def extract_residual_activations(
        self,
        prompts: List[str],
        max_positions: int = 7
    ) -> Dict[Tuple[int, int], t.Tensor]:
        """Extracts and averages activations for a batch of prompts.

        Response-contrast strings (prompt COMPLETION_SEP completion) are run as
        [templated prompt | completion]; position -1 is then the mean over the
        completion's tokens and -k the k-th token from the completion's end."""
        if any(COMPLETION_SEP in p for p in prompts):
            if not all(COMPLETION_SEP in p for p in prompts):
                raise ValueError("Mixed prompt-only and prompt+completion strings in one extraction")
            return self._extract_over_completions(prompts, max_positions)
        all_activations = {}

        batch = self.prompt_formatter.format_batch(prompts)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)

        batch_size, seq_len = input_ids.shape
        # activations are read at `true_len + pos_idx`, valid only under right padding
        assert_right_padded(attention_mask)
        true_lengths = attention_mask.sum(dim=1)

        with t.no_grad():
            activations_by_layer = {}
            def make_hook(layer_idx):
                def hook(module, args):
                    # Pre-hook: args[0] is hidden_states (residual stream entering the layer)
                    # Matches Arditi: captures residual stream *before* this layer processes it
                    hidden_states = args[0]
                    activations_by_layer[layer_idx] = hidden_states.clone().cpu()
                return hook

            hooks = [layer.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(self.transformer_layers)]

            try:
                self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            finally:
                for hook in hooks: hook.remove()

            for layer_idx, layer_acts_batch in activations_by_layer.items():
                for i in range(batch_size):
                    true_len = true_lengths[i].item()
                    for pos_idx in range(-1, -min(max_positions, true_len) - 1, -1):
                        key = (layer_idx, pos_idx)
                        # Right padding: real tokens occupy 0..true_len-1, so true_len + pos_idx
                        # is the pos_idx-th real token from the end ([batch, seq, d_model] tensor).
                        # Cast to float64 for numerical stability when averaging
                        # (bf16 quantization noise accumulates over ~100 samples)
                        activation = layer_acts_batch[i, true_len + pos_idx, :].to(t.float64)

                        if key not in all_activations:
                            all_activations[key] = []
                        all_activations[key].append(activation)

        # Mean computed in float64 for precision; stays float64 for downstream subtraction
        return {key: t.stack(acts).mean(dim=0) for key, acts in all_activations.items()}


    def _extract_over_completions(self, strings: List[str], max_positions: int) -> Dict[Tuple[int, int], t.Tensor]:
        pairs = [s.split(COMPLETION_SEP, 1) for s in strings]
        batch = self.prompt_formatter.format_with_completions([p for p, _ in pairs], [c for _, c in pairs])
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)
        assert_right_padded(attention_mask)
        scored = (batch['labels'] != -100)                      # completion tokens, per row
        all_activations: Dict[Tuple[int, int], List[t.Tensor]] = {}
        with t.no_grad():
            captured = {}

            def make_hook(layer_idx):
                def hook(module, args):
                    captured[layer_idx] = args[0].clone().cpu()
                return hook
            hooks = [layer.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(self.transformer_layers)]
            try:
                self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            finally:
                for hook in hooks:
                    hook.remove()
        for layer_idx, acts in captured.items():
            for i in range(acts.shape[0]):
                idx = scored[i].nonzero().flatten()
                if len(idx) == 0:
                    continue
                rows = acts[i, idx, :].to(t.float64)
                all_activations.setdefault((layer_idx, -1), []).append(rows.mean(0))
                for k in range(2, max_positions + 1):
                    if len(idx) >= k:
                        all_activations.setdefault((layer_idx, -k), []).append(rows[-k])
        return {key: t.stack(v).mean(dim=0) for key, v in all_activations.items()}
