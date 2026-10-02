import torch as t
from typing import List, Optional, Union
from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging

from datatypes import DirectionVector
from orthogonalize import orthonormal_basis

logger = logging.getLogger(__name__)


def get_sublayers(block):
    """Returns (attention_sublayer, mlp_sublayer) for a transformer block."""
    if hasattr(block, 'self_attn') and hasattr(block, 'mlp'):
        return block.self_attn, block.mlp  # Llama, Gemma, Qwen2
    elif hasattr(block, 'attn') and hasattr(block, 'mlp'):
        return block.attn, block.mlp  # GPT-2, DialoGPT
    raise AttributeError(f"Could not identify sublayers for block {block.__class__.__name__}.")


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
        return get_sublayers(block)

    def apply_direction_intervention(
        self,
        direction: Union[DirectionVector, t.Tensor],
        intervention_type: str = "add",
        strength: float = 1.0,
        layers: Optional[List[int]] = None
    ):
        """Applies a direction intervention to specified model layers.

        `direction` is a DirectionVector or a raw tensor: a [d] vector, or (for
        "ablate" only) a [k, d] stack of directions whose whole span is removed in one
        order-independent projection, x - (x Qᵀ) Q, with Q = `orthogonalize.
        orthonormal_basis` of the rows (which raises on linearly dependent rows).
        A [1, d] stack is the single-direction case and takes the same code path."""
        if layers is None:
            layers = list(range(len(self.transformer_layers)))

        vector = direction.vector if isinstance(direction, DirectionVector) else direction
        if vector.dim() == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.dim() == 2:
            if intervention_type != "ablate":
                raise ValueError(f"A [k, d] stack of directions only supports 'ablate', got {intervention_type!r}")
            basis = orthonormal_basis(vector).to(self.model.dtype)  # [k, d]

            def project_out(hidden_states):
                q = basis.to(hidden_states.device)
                return hidden_states - t.matmul(t.matmul(hidden_states, q.T), q)
        elif vector.dim() == 1:
            norm = t.norm(vector)
            if intervention_type == "ablate" and not (norm > 0):
                # 0/0 would make every hidden state NaN; a zero direction (e.g. layer-0
                # difference-in-means at a shared template token) can't be ablated.
                raise ValueError(f"Cannot ablate a direction with norm {float(norm)}")
            unit_dir = (vector / norm).to(self.model.dtype)
            raw_dir = vector.to(self.model.dtype)

            def project_out(hidden_states):
                device_unit_dir = unit_dir.to(hidden_states.device)
                projection = t.sum(hidden_states * device_unit_dir, dim=-1, keepdim=True)
                return hidden_states - projection * device_unit_dir
        else:
            raise ValueError(f"direction must be [d] or [k, d], got shape {tuple(vector.shape)}")

        def make_block_pre_hook(intervention_type, strength):
            """Pre-hook on transformer block: modifies residual stream before the layer processes it."""
            def hook(module, args):
                hidden_states = args[0]

                if intervention_type == "add":
                    device_raw_dir = raw_dir.to(hidden_states.device)
                    modified_states = hidden_states + strength * device_raw_dir
                elif intervention_type == "subtract":
                    device_raw_dir = raw_dir.to(hidden_states.device)
                    modified_states = hidden_states - strength * device_raw_dir
                elif intervention_type == "ablate":
                    modified_states = project_out(hidden_states)
                else:
                    raise ValueError(f"Unknown intervention type: {intervention_type}")

                return (modified_states,) + args[1:]
            return hook

        def make_sublayer_post_hook():
            """Post-hook on attn/mlp sublayer: projects out the direction(s) from sublayer
            output. Prevents the sublayer from re-injecting them into the residual stream."""
            def hook(module, input, output):
                is_tuple_output = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple_output else output
                modified_states = project_out(hidden_states)

                if is_tuple_output:
                    return (modified_states,) + output[1:]
                else:
                    return modified_states
            return hook

        block_hook_fn = make_block_pre_hook(intervention_type, strength)
        try:
            for layer_idx in layers:
                if 0 <= layer_idx < len(self.transformer_layers):
                    block = self.transformer_layers[layer_idx]
                    hook = block.register_forward_pre_hook(block_hook_fn)
                    self.intervention_hooks.append(hook)

                    # For ablation: also hook sublayer outputs to prevent re-injection
                    if intervention_type == "ablate":
                        attn, mlp = self._get_sublayers(block)
                        sublayer_hook_fn = make_sublayer_post_hook()
                        self.intervention_hooks.append(attn.register_forward_hook(sublayer_hook_fn))
                        self.intervention_hooks.append(mlp.register_forward_hook(sublayer_hook_fn))
        except Exception:
            # Registration failed partway through the loop — make it atomic by
            # removing everything registered so far (including from this call)
            # and re-raising so callers see the failure.
            self.clear_interventions()
            raise

    @contextmanager
    def intervened(
        self,
        direction: Union[DirectionVector, t.Tensor],
        intervention_type: str = "add",
        strength: float = 1.0,
        layers: Optional[List[int]] = None
    ):
        """Context manager: `apply_direction_intervention(...)` on entry,
        `clear_interventions()` on exit (normal or exception). Use this instead of
        hand-rolling apply/try/finally — a missed `finally` leaks hooks into every
        later forward pass.

        Like `clear_interventions()`, exit removes *all* registered hooks, not only
        the ones this block added, so don't nest it inside other intervention hooks."""
        self.apply_direction_intervention(direction, intervention_type, strength, layers=layers)
        try:
            yield self
        finally:
            self.clear_interventions()

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
