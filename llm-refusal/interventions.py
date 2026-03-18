import torch as t
from typing import List, Optional
from abc import ABC, abstractmethod
import logging

from datatypes import DirectionVector

logger = logging.getLogger(__name__)


class ModelInterventionApplier:
    """Handles the application and removal of interventions in a model's forward pass using hooks."""

    def __init__(self, model):
        self.model = model
        self.intervention_hooks = []
        self.transformer_layers = self._get_transformer_layers()
        logger.info(f"Identified {len(self.transformer_layers)} transformer layers.")

    def _get_transformer_layers(self):
        """Dynamically identifies the list of transformer layers in various model architectures."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # Llama, Gemma, Qwen2
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h  # GPT-2, DialoGPT
        raise AttributeError(f"Could not automatically identify transformer layers for model {self.model.__class__.__name__}.")

    def _get_sublayers(self, block):
        """Returns (attention_sublayer, mlp_sublayer) for a transformer block."""
        if hasattr(block, 'self_attn') and hasattr(block, 'mlp'):
            return block.self_attn, block.mlp  # Llama, Gemma, Qwen2
        elif hasattr(block, 'attn') and hasattr(block, 'mlp'):
            return block.attn, block.mlp  # GPT-2, DialoGPT
        raise AttributeError(f"Could not identify sublayers for block {block.__class__.__name__}.")

    def apply_direction_intervention(
        self,
        direction: DirectionVector,
        intervention_type: str = "add",
        strength: float = 1.0,
        layers: Optional[List[int]] = None
    ):
        """Applies a direction intervention to specified model layers."""
        if layers is None:
            layers = list(range(len(self.transformer_layers)))

        unit_dir = direction.unit.to(self.model.dtype)
        raw_dir = direction.vector.to(self.model.dtype)

        def make_block_pre_hook(intervention_type, strength, unit_dir, raw_dir):
            """Pre-hook on transformer block: modifies residual stream before the layer processes it."""
            def hook(module, args):
                hidden_states = args[0]
                device_unit_dir = unit_dir.to(hidden_states.device)

                if intervention_type == "add":
                    device_raw_dir = raw_dir.to(hidden_states.device)
                    modified_states = hidden_states + strength * device_raw_dir
                elif intervention_type == "subtract":
                    device_raw_dir = raw_dir.to(hidden_states.device)
                    modified_states = hidden_states - strength * device_raw_dir
                elif intervention_type == "ablate":
                    projection = t.sum(hidden_states * device_unit_dir, dim=-1, keepdim=True)
                    modified_states = hidden_states - projection * device_unit_dir
                else:
                    raise ValueError(f"Unknown intervention type: {intervention_type}")

                return (modified_states,) + args[1:]
            return hook

        def make_sublayer_post_hook(unit_dir):
            """Post-hook on attn/mlp sublayer: projects out direction from sublayer output.
            Prevents the sublayer from re-injecting the direction into the residual stream."""
            def hook(module, input, output):
                is_tuple_output = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple_output else output
                device_unit_dir = unit_dir.to(hidden_states.device)

                projection = t.sum(hidden_states * device_unit_dir, dim=-1, keepdim=True)
                modified_states = hidden_states - projection * device_unit_dir

                if is_tuple_output:
                    return (modified_states,) + output[1:]
                else:
                    return modified_states
            return hook

        block_hook_fn = make_block_pre_hook(intervention_type, strength, unit_dir, raw_dir)
        for layer_idx in layers:
            if 0 <= layer_idx < len(self.transformer_layers):
                block = self.transformer_layers[layer_idx]
                hook = block.register_forward_pre_hook(block_hook_fn)
                self.intervention_hooks.append(hook)

                # For ablation: also hook sublayer outputs to prevent re-injection
                if intervention_type == "ablate":
                    attn, mlp = self._get_sublayers(block)
                    sublayer_hook_fn = make_sublayer_post_hook(unit_dir)
                    self.intervention_hooks.append(attn.register_forward_hook(sublayer_hook_fn))
                    self.intervention_hooks.append(mlp.register_forward_hook(sublayer_hook_fn))

    def clear_interventions(self):
        """Removes all active intervention hooks."""
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks = []


class InterventionStrategy(ABC):
    """Abstract base class for defining how an intervention is applied."""
    def __init__(self, intervention_applier: ModelInterventionApplier):
        self.intervention_applier = intervention_applier

    @abstractmethod
    def apply_intervention(self, direction_vector: DirectionVector) -> None:
        """Applies the intervention to the model."""
        pass

    def clear_intervention(self) -> None:
        """Clears any applied interventions."""
        self.intervention_applier.clear_interventions()


class GlobalInterventionStrategy(InterventionStrategy):
    """Applies the intervention to all transformer layers."""
    def apply_intervention(self, direction_vector: DirectionVector) -> None:
        self.intervention_applier.apply_direction_intervention(
            direction_vector, intervention_type="add", strength=1.0, layers=None
        )


class LayerSpecificInterventionStrategy(InterventionStrategy):
    """Applies the intervention only to the layer where the direction was discovered."""
    def apply_intervention(self, direction_vector: DirectionVector) -> None:
        self.intervention_applier.apply_direction_intervention(
            direction_vector, intervention_type="add", strength=1.0, layers=[direction_vector.layer]
        )
