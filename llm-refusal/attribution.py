"""Attribution analysis: decompose which attention heads and MLP sublayers
write the refusal direction into the residual stream.

For each transformer layer, the residual stream update is:
    residual_out = residual_in + attn_output + mlp_output

The attention output is the sum of individual head contributions through o_proj:
    attn_output = o_proj(concat(head_0, head_1, ..., head_n))
               = sum_i(W_o[:, i*hd:(i+1)*hd] @ head_i)    (o_proj has no bias in Llama/Qwen)

We project each component's output onto the refusal direction unit vector to measure
how much it contributes to the refusal signal. Contrastive attribution (harmful - benign)
isolates components that specifically respond to harmfulness vs. generic model behavior.
"""

import torch as t
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from formatting import ChatPromptFormatter, last_real_token_indices, assert_right_padded
from datatypes import DirectionVector
from interventions import get_sublayers

logger = logging.getLogger(__name__)


@dataclass
class LayerAttribution:
    """Attribution scores for a single layer."""
    layer_idx: int
    attn_projection: float
    mlp_projection: float
    total_projection: float  # attn + mlp


@dataclass
class HeadAttribution:
    """Per-head attribution scores within a layer."""
    layer_idx: int
    head_projections: Dict[int, float]  # head_idx -> projection onto refusal direction
    mlp_projection: float

    @property
    def top_heads(self) -> List[Tuple[int, float]]:
        """Heads sorted by absolute projection magnitude (descending)."""
        return sorted(self.head_projections.items(), key=lambda x: abs(x[1]), reverse=True)


@dataclass
class CircuitComponent:
    """A single component in the refusal circuit."""
    layer: int
    component_type: str  # "head" or "mlp"
    head_idx: Optional[int] = None
    attribution: float = 0.0       # projection onto refusal direction (from attribution)
    causal_effect: Optional[float] = None  # change in refusal log-odds when ablated


@dataclass
class CircuitResult:
    """Full results from circuit analysis on one model."""
    model_name: str
    direction_layer: int
    direction_pos: int
    num_layers: int
    num_heads: int
    layer_attributions: List[LayerAttribution] = field(default_factory=list)
    contrastive_layer_attributions: List[LayerAttribution] = field(default_factory=list)
    head_attributions: List[HeadAttribution] = field(default_factory=list)
    contrastive_head_attributions: List[HeadAttribution] = field(default_factory=list)
    top_components: List[CircuitComponent] = field(default_factory=list)
    verified_components: List[CircuitComponent] = field(default_factory=list)


class AttributionAnalyzer:
    """Decomposes which attention heads and MLP sublayers write the refusal direction."""

    def __init__(self, model, tokenizer, transformer_layers, prompt_formatter: ChatPromptFormatter):
        self.model = model
        self.tokenizer = tokenizer
        self.transformer_layers = transformer_layers
        self.prompt_formatter = prompt_formatter
        self.device = model.device

    _get_sublayers = staticmethod(get_sublayers)

    def _get_o_proj(self, attn_module):
        if hasattr(attn_module, 'o_proj'):
            return attn_module.o_proj
        elif hasattr(attn_module, 'c_proj'):
            return attn_module.c_proj
        raise AttributeError(f"Could not find output projection in {attn_module.__class__.__name__}")

    def _get_num_heads(self, attn_module):
        for attr in ('num_heads', 'num_attention_heads'):
            if hasattr(attn_module, attr):
                return getattr(attn_module, attr)
        # Qwen2/Llama store it in config
        if hasattr(attn_module, 'config') and hasattr(attn_module.config, 'num_attention_heads'):
            return attn_module.config.num_attention_heads
        raise AttributeError(f"Could not find num_heads in {attn_module.__class__.__name__}")

    def _get_head_dim(self, attn_module):
        if hasattr(attn_module, 'head_dim'):
            return attn_module.head_dim
        return self._get_o_proj(attn_module).in_features // self._get_num_heads(attn_module)

    # ── Layer-level attribution ─────────────────────────────────────

    def compute_layer_attributions(
        self,
        prompts: List[str],
        direction: DirectionVector,
    ) -> List[LayerAttribution]:
        """For each layer, project attn and MLP outputs onto the refusal direction.

        Uses the direction's position_index to extract the right token position.
        Returns average projection across all prompts.
        """
        unit_dir = direction.unit.float().to(self.device)
        pos_idx = direction.position_index

        batch = self.prompt_formatter.format_batch(prompts)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)
        assert_right_padded(attention_mask)
        true_lengths = attention_mask.sum(dim=1)

        attn_outputs = {}
        mlp_outputs = {}
        hooks = []

        for layer_idx, block in enumerate(self.transformer_layers):
            attn, mlp = self._get_sublayers(block)

            def make_attn_hook(idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    attn_outputs[idx] = hidden.detach()
                return hook

            def make_mlp_hook(idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    mlp_outputs[idx] = hidden.detach()
                return hook

            hooks.append(attn.register_forward_hook(make_attn_hook(layer_idx)))
            hooks.append(mlp.register_forward_hook(make_mlp_hook(layer_idx)))

        try:
            with t.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        finally:
            for h in hooks:
                h.remove()

        attributions = []
        for layer_idx in range(len(self.transformer_layers)):
            attn_projs = []
            mlp_projs = []
            for i in range(len(prompts)):
                actual_pos = true_lengths[i].item() + pos_idx
                attn_vec = attn_outputs[layer_idx][i, actual_pos, :].float()
                mlp_vec = mlp_outputs[layer_idx][i, actual_pos, :].float()
                attn_projs.append(t.dot(attn_vec, unit_dir).item())
                mlp_projs.append(t.dot(mlp_vec, unit_dir).item())

            avg_attn = float(np.mean(attn_projs))
            avg_mlp = float(np.mean(mlp_projs))
            attributions.append(LayerAttribution(
                layer_idx=layer_idx,
                attn_projection=avg_attn,
                mlp_projection=avg_mlp,
                total_projection=avg_attn + avg_mlp,
            ))

        return attributions

    def compute_contrastive_layer_attributions(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        direction: DirectionVector,
    ) -> Tuple[List[LayerAttribution], List[LayerAttribution], List[LayerAttribution]]:
        """Attribution on harmful prompts, benign prompts, and the contrastive difference.

        Returns (positive_attr, negative_attr, contrastive_attr).
        Contrastive = positive - negative, isolating refusal-specific signal.
        """
        pos_attr = self.compute_layer_attributions(positive_prompts, direction)
        neg_attr = self.compute_layer_attributions(negative_prompts, direction)

        contrastive = []
        for p, n in zip(pos_attr, neg_attr):
            contrastive.append(LayerAttribution(
                layer_idx=p.layer_idx,
                attn_projection=p.attn_projection - n.attn_projection,
                mlp_projection=p.mlp_projection - n.mlp_projection,
                total_projection=p.total_projection - n.total_projection,
            ))

        return pos_attr, neg_attr, contrastive

    # ── Head-level attribution ──────────────────────────────────────

    def compute_head_attributions(
        self,
        prompts: List[str],
        direction: DirectionVector,
        layers: Optional[List[int]] = None,
    ) -> List[HeadAttribution]:
        """Decompose attention into per-head contributions to the refusal direction.

        Hooks o_proj's input (concatenated head outputs before output projection),
        then decomposes: head_i_contribution = W_o[:, i*hd:(i+1)*hd] @ head_i_values.
        """
        if layers is None:
            layers = list(range(len(self.transformer_layers)))

        unit_dir = direction.unit.float().to(self.device)
        pos_idx = direction.position_index

        batch = self.prompt_formatter.format_batch(prompts)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)
        assert_right_padded(attention_mask)
        true_lengths = attention_mask.sum(dim=1)

        o_proj_inputs = {}
        mlp_outputs_dict = {}
        hooks = []

        for layer_idx in layers:
            block = self.transformer_layers[layer_idx]
            attn, mlp = self._get_sublayers(block)
            o_proj = self._get_o_proj(attn)

            def make_o_proj_pre_hook(idx):
                def hook(module, args):
                    o_proj_inputs[idx] = args[0].detach()
                return hook

            def make_mlp_hook(idx):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    mlp_outputs_dict[idx] = hidden.detach()
                return hook

            hooks.append(o_proj.register_forward_pre_hook(make_o_proj_pre_hook(layer_idx)))
            hooks.append(mlp.register_forward_hook(make_mlp_hook(layer_idx)))

        try:
            with t.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        finally:
            for h in hooks:
                h.remove()

        results = []
        for layer_idx in layers:
            block = self.transformer_layers[layer_idx]
            attn, _ = self._get_sublayers(block)
            o_proj = self._get_o_proj(attn)
            num_heads = self._get_num_heads(attn)
            head_dim = self._get_head_dim(attn)
            W_o = o_proj.weight.float()  # [hidden_size, num_heads * head_dim]

            head_proj_accum = {h: [] for h in range(num_heads)}
            mlp_proj_accum = []

            for i in range(len(prompts)):
                actual_pos = true_lengths[i].item() + pos_idx
                concat_heads = o_proj_inputs[layer_idx][i, actual_pos, :].float()

                for h in range(num_heads):
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    head_contribution = W_o[:, start:end] @ concat_heads[start:end]
                    proj = t.dot(head_contribution, unit_dir).item()
                    head_proj_accum[h].append(proj)

                mlp_vec = mlp_outputs_dict[layer_idx][i, actual_pos, :].float()
                mlp_proj_accum.append(t.dot(mlp_vec, unit_dir).item())

            head_projs = {h: float(np.mean(vals)) for h, vals in head_proj_accum.items()}
            mlp_proj = float(np.mean(mlp_proj_accum))

            # If o_proj has bias, distribute evenly (rare — Llama/Qwen don't use it)
            if o_proj.bias is not None:
                bias_proj = t.dot(o_proj.bias.float(), unit_dir).item() / num_heads
                head_projs = {h: v + bias_proj for h, v in head_projs.items()}

            results.append(HeadAttribution(
                layer_idx=layer_idx,
                head_projections=head_projs,
                mlp_projection=mlp_proj,
            ))

        return results

    def compute_contrastive_head_attributions(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        direction: DirectionVector,
        layers: Optional[List[int]] = None,
    ) -> Tuple[List[HeadAttribution], List[HeadAttribution], List[HeadAttribution]]:
        """Head-level attribution on harmful, benign, and the contrastive difference."""
        pos_attr = self.compute_head_attributions(positive_prompts, direction, layers)
        neg_attr = self.compute_head_attributions(negative_prompts, direction, layers)

        contrastive = []
        for p, n in zip(pos_attr, neg_attr):
            diff_heads = {h: p.head_projections[h] - n.head_projections[h]
                          for h in p.head_projections}
            contrastive.append(HeadAttribution(
                layer_idx=p.layer_idx,
                head_projections=diff_heads,
                mlp_projection=p.mlp_projection - n.mlp_projection,
            ))

        return pos_attr, neg_attr, contrastive

    # ── Causal verification ─────────────────────────────────────────

    def verify_circuit_causally(
        self,
        prompts: List[str],
        direction: DirectionVector,
        components: List[CircuitComponent],
    ) -> List[CircuitComponent]:
        """Ablate each component's refusal-direction contribution and measure log-odds change.

        For heads: hooks o_proj input, removes the component of head_h's values that
        would project onto the refusal direction after o_proj.
        For MLP: removes MLP output's projection onto the refusal direction.
        """
        from scoring import LogOddsMetric
        from concept import DEFAULT_REFUSAL_TOKENS

        metric = LogOddsMetric(self.tokenizer, DEFAULT_REFUSAL_TOKENS)
        unit_dir = direction.unit.float().to(self.device)

        batch = self.prompt_formatter.format_batch(prompts)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)
        last_indices = last_real_token_indices(attention_mask)

        # Baseline refusal log-odds
        with t.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        baseline_scores = []
        for i in range(len(prompts)):
            logits = outputs.logits[i, last_indices[i], :]
            baseline_scores.append(metric.compute_log_odds(logits))
        baseline_avg = float(np.nanmean(baseline_scores))
        logger.info(f"  Baseline refusal log-odds: {baseline_avg:.4f}")

        verified = []
        for comp in components:
            block = self.transformer_layers[comp.layer]
            attn, mlp = self._get_sublayers(block)
            hooks = []

            if comp.component_type == "head":
                o_proj = self._get_o_proj(attn)
                head_dim = self._get_head_dim(attn)
                W_o = o_proj.weight.float()
                start = comp.head_idx * head_dim
                end = (comp.head_idx + 1) * head_dim

                # Precompute: direction in head-value space that produces refusal direction after o_proj
                # proj_weight = W_o[:, start:end]^T @ unit_dir  (shape: [head_dim])
                proj_weight = (W_o[:, start:end].T @ unit_dir).to(self.device)
                proj_weight_norm = t.norm(proj_weight)

                def make_head_ablation_hook(s, e, pw, pw_norm):
                    def hook(module, args):
                        x = args[0].clone()
                        head_vals = x[:, :, s:e].float()
                        # Remove component along proj_weight direction
                        pw_unit = pw / pw_norm
                        proj_mag = (head_vals * pw_unit).sum(dim=-1, keepdim=True)
                        x[:, :, s:e] = (head_vals - proj_mag * pw_unit).to(x.dtype)
                        return (x,) + args[1:]
                    return hook

                hooks.append(o_proj.register_forward_pre_hook(
                    make_head_ablation_hook(start, end, proj_weight, proj_weight_norm)
                ))

            elif comp.component_type == "mlp":
                def make_mlp_ablation_hook(ud):
                    def hook(module, input, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        proj = (hidden.float() * ud).sum(dim=-1, keepdim=True)
                        modified = (hidden.float() - proj * ud).to(hidden.dtype)
                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        return modified
                    return hook

                hooks.append(mlp.register_forward_hook(make_mlp_ablation_hook(unit_dir)))

            try:
                with t.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
                ablated_scores = []
                for i in range(len(prompts)):
                    logits = outputs.logits[i, last_indices[i], :]
                    ablated_scores.append(metric.compute_log_odds(logits))
                ablated_avg = float(np.nanmean(ablated_scores))
            finally:
                for h in hooks:
                    h.remove()

            comp_verified = CircuitComponent(
                layer=comp.layer,
                component_type=comp.component_type,
                head_idx=comp.head_idx,
                attribution=comp.attribution,
                causal_effect=ablated_avg - baseline_avg,
            )

            label = f"L{comp.layer}.{'H' + str(comp.head_idx) if comp.component_type == 'head' else 'MLP'}"
            logger.info(f"  Ablate {label}: {baseline_avg:.4f} -> {ablated_avg:.4f} (delta={comp_verified.causal_effect:+.4f})")
            verified.append(comp_verified)

        return verified

    # ── Convenience: full pipeline ──────────────────────────────────

    def find_refusal_circuit(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        direction: DirectionVector,
        top_layers_k: int = 5,
        top_components_k: int = 15,
    ) -> CircuitResult:
        """Full pipeline: layer attribution -> head attribution -> causal verification.

        Args:
            positive_prompts: Harmful prompts (model refuses these).
            negative_prompts: Benign prompts (model complies).
            direction: The refusal direction vector.
            top_layers_k: Number of top layers to decompose into heads.
            top_components_k: Number of top components to verify causally.
        """
        model_name = self.tokenizer.name_or_path
        attn_0, _ = self._get_sublayers(self.transformer_layers[0])
        num_heads = self._get_num_heads(attn_0)
        num_layers = len(self.transformer_layers)

        result = CircuitResult(
            model_name=model_name,
            direction_layer=direction.layer,
            direction_pos=direction.position_index,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Step 1: Layer-level attribution (contrastive)
        logger.info("Step 1: Computing layer-level attributions...")
        pos_layer, neg_layer, contrast_layer = self.compute_contrastive_layer_attributions(
            positive_prompts, negative_prompts, direction
        )
        result.layer_attributions = pos_layer
        result.contrastive_layer_attributions = contrast_layer

        # Identify top layers by absolute contrastive total projection
        sorted_layers = sorted(contrast_layer, key=lambda a: abs(a.total_projection), reverse=True)
        top_layer_idxs = [a.layer_idx for a in sorted_layers[:top_layers_k]]
        logger.info(f"  Top {top_layers_k} layers by contrastive attribution: {top_layer_idxs}")
        for a in sorted_layers[:top_layers_k]:
            logger.info(f"    L{a.layer_idx}: attn={a.attn_projection:+.4f}, mlp={a.mlp_projection:+.4f}, total={a.total_projection:+.4f}")

        # Step 2: Head-level attribution on top layers
        logger.info(f"Step 2: Computing head-level attributions for layers {top_layer_idxs}...")
        pos_head, neg_head, contrast_head = self.compute_contrastive_head_attributions(
            positive_prompts, negative_prompts, direction, layers=top_layer_idxs
        )
        result.head_attributions = pos_head
        result.contrastive_head_attributions = contrast_head

        # Collect all components (heads + MLPs) from top layers, rank by contrastive attribution
        all_components = []
        for ha in contrast_head:
            for h_idx, proj in ha.head_projections.items():
                all_components.append(CircuitComponent(
                    layer=ha.layer_idx, component_type="head",
                    head_idx=h_idx, attribution=proj,
                ))
            all_components.append(CircuitComponent(
                layer=ha.layer_idx, component_type="mlp",
                head_idx=None, attribution=ha.mlp_projection,
            ))

        all_components.sort(key=lambda c: abs(c.attribution), reverse=True)
        top_components = all_components[:top_components_k]
        result.top_components = top_components

        logger.info(f"  Top {top_components_k} components by contrastive attribution:")
        for c in top_components:
            label = f"L{c.layer}.{'H' + str(c.head_idx) if c.component_type == 'head' else 'MLP'}"
            logger.info(f"    {label}: attribution={c.attribution:+.4f}")

        # Step 3: Causal verification on top components
        logger.info(f"Step 3: Causal verification on top {top_components_k} components...")
        result.verified_components = self.verify_circuit_causally(
            positive_prompts, direction, top_components
        )

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"REFUSAL CIRCUIT SUMMARY — {model_name}")
        logger.info(f"Direction: layer={direction.layer}, pos={direction.position_index}")
        logger.info(f"Model: {num_layers} layers, {num_heads} heads/layer")
        logger.info("=" * 60)
        logger.info(f"{'Component':<16} {'Attribution':>12} {'Causal Δ':>12}")
        logger.info("-" * 40)
        for c in sorted(result.verified_components, key=lambda c: c.causal_effect or 0):
            label = f"L{c.layer}.{'H' + str(c.head_idx) if c.component_type == 'head' else 'MLP'}"
            causal_str = f"{c.causal_effect:+.4f}" if c.causal_effect is not None else "N/A"
            logger.info(f"{label:<16} {c.attribution:>+12.4f} {causal_str:>12}")
        logger.info("=" * 60)

        return result
