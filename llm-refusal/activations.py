import torch as t
from typing import List, Dict, Tuple
import logging

from formatting import ChatPromptFormatter

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extracts residual stream activations from a model, using batching.
    Corrected to handle various model output formats (tuple vs. tensor)."""
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
        """Extracts and averages activations for a batch of prompts."""
        all_activations = {}

        batch = self.prompt_formatter.format_batch(prompts)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)

        batch_size, seq_len = input_ids.shape
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
                        # This indexing is now correct because layer_acts_batch is guaranteed to be 3D
                        # Cast to float64 for numerical stability when averaging
                        # (bf16 quantization noise accumulates over ~100 samples)
                        activation = layer_acts_batch[i, true_len + pos_idx, :].to(t.float64)

                        if key not in all_activations:
                            all_activations[key] = []
                        all_activations[key].append(activation)

        # Mean computed in float64 for precision; stays float64 for downstream subtraction
        return {key: t.stack(acts).mean(dim=0) for key, acts in all_activations.items()}
