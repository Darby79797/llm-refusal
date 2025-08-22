import numpy as np 
import torch as t

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PromptData:
    prompts: List[str]
    labels: List[bool]  # True for positive examples, False for negative.
    
    def train_val_split(self, test_size: float = 0.2, random_state: int = 39):
        train_prompts, val_prompts, train_labels, val_labels = train_test_split(
            self.prompts, self.labels, test_size=test_size, random_state=random_state
        )
        return (
            PromptData(train_prompts, train_labels),
            PromptData(val_prompts, val_labels)
        )
    
@dataclass 
class DirectionVector:
    vector: t.Tensor
    layer: int
    position_index: int  # a negative index, from end of prompt
    score: float  # evaluation metric score
    
    @property
    def unit(self) -> t.Tensor:
        return self.vector / t.norm(self.vector)
    

class ActivationExtractor:
    # for getting activations out of (possibly large) models, intended to compute difference_in_means.
    # TransformerLens definitely has nicer ways of doing this than re-implementing for small models.
    # unsure about nnsight. 
    # current plan: after we have successfully computed difference-in-means locally on a ~2B or ~7B model, consider learning nnsight
    # (meta-plan: just ship something)
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_residual_activations(
        self, 
        prompts: List[str], 
        max_positions: int = 7
    ) -> Dict[Tuple[int, int], t.Tensor]:
        """
        Extract activations at different points in the residual stream, averaged across prompts, for tail token positions.
        (prompts need not be the same length)
        
        Args:
            prompts: List of input prompts
            max_positions: Maximum number of positions from end to consider
            
        Returns:
            Dict mapping (layer, position_idx) -> averaged activations
        """
        # local extended comments: we design this to not require
        all_activations = {}
        
        for prompt in prompts:
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            seq_len = tokens.input_ids.shape[1]
            
            with t.no_grad():
                # a pytorch hook, to collect activations
                activations_by_layer = {}
                
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                        else:
                            hidden_states = output
                        activations_by_layer[layer_idx] = hidden_states.clone()
                    return hook
                
                # Register hooks on transformer blocks
                hooks = []
                for i, layer in enumerate(self.model.transformer.h):  # THIS MIGHT NOT WORK?
                    hook = layer.register_forward_hook(make_hook(i))
                    hooks.append(hook)
                
                # Forward pass
                _ = self.model(**tokens)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Extract activations at different positions from end
                for layer_idx, layer_acts in activations_by_layer.items():
                    for pos_idx in range(-1, -max_positions - 1, -1):
                        if abs(pos_idx) <= seq_len:
                            key = (layer_idx, pos_idx)
                            activation = layer_acts[0, pos_idx, :]  # [batch=1, seq, hidden]
                            
                            if key not in all_activations:
                                all_activations[key] = []
                            all_activations[key].append(activation)
        
        # Average activations across prompts
        averaged_activations = {}
        for key, acts in all_activations.items():
            averaged_activations[key] = t.stack(acts).mean(dim=0)
            
        return averaged_activations
    
class DifferenceInMeans:
    """Finds the direction (at all layers of the residual stream, for tail token positions) that is the difference-in-mean activations between positive and negative prompts."""
    
    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor
        
    def compute_difference_vectors(
        self, 
        train_data: PromptData,
        max_positions: int = 7
    ) -> Dict[Tuple[int, int], t.Tensor]:
        """
        Compute difference-in-means vectors for all layer/position combinations.
        
        Returns:
            Dict mapping (layer, position_idx) -> difference vector
        """
        positive_prompts = [p for p, label in zip(train_data.prompts, train_data.labels) if label]
        negative_prompts = [p for p, label in zip(train_data.prompts, train_data.labels) if not label]
        
        logger.info(f"Computing activations for {len(positive_prompts)} positive and {len(negative_prompts)} negative prompts")
        
        pos_activations = self.extractor.extract_residual_activations(positive_prompts, max_positions)
        neg_activations = self.extractor.extract_residual_activations(negative_prompts, max_positions)
        
        difference_vectors = {}
        for key in pos_activations.keys():
            if key in neg_activations:
                diff_vec = pos_activations[key] - neg_activations[key]
                difference_vectors[key] = diff_vec
                
        return difference_vectors
    
    
# in general, I'm not sure about Claude's approach to use abstract classes at all. Again, consider later.
class DirectionEvaluator(ABC):
    """Abstract base class for evaluating how well a difference-in-means on prompt data (train) """
    
    @abstractmethod
    def evaluate_direction(
        self, 
        direction_vector: DirectionVector, 
        val_data: PromptData
    ) -> float:
        """Evaluate how well a direction captures the target concept."""
        pass

class LogOddsMetric:
    """Handles log-odds computation from model logits."""
    # why is this a class? consider just making it a function with more inputs when I have some time.
    def __init__(self, tokenizer, target_tokens: List[str]):
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.target_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] 
                                for token in target_tokens]
    
    def compute_log_odds(self, logits: t.Tensor) -> float:
        """
        Compute log-odds of target tokens from model logits.
        
        Args:
            logits: Raw logits from model [vocab_size]
            
        Returns:
            Log-odds: log(p / (1-p)) where p is probability of target tokens
        """
        # Get logits for target tokens
        target_logits = logits[self.target_token_ids]  # [num_target_tokens]
        
        # Compute probabilities using logsumexp for numerical stability
        # P(target) = sum(exp(target_logits)) / sum(exp(all_logits))
        target_log_sum_exp = t.logsumexp(target_logits, dim=0)
        all_log_sum_exp = t.logsumexp(logits, dim=0)
        
        # log P(target) = log_sum_exp(target_logits) - log_sum_exp(all_logits)
        log_p_target = target_log_sum_exp - all_log_sum_exp
        
        # log P(not target) = log(1 - P(target))
        p_target = t.exp(log_p_target)
        log_p_not_target = t.log(1 - p_target + 1e-8)  # Add epsilon for numerical stability
        
        # Log-odds = log(P(target) / P(not target))
        log_odds = log_p_target - log_p_not_target
        
        return log_odds.item()

class InterventionStrategy(ABC):
    """Abstract base class for different intervention strategies:
    Adding/subtracting a direction from the residual stream
    or ablating a direction from the residual stream,
    at one, some or all layers."""
    
    def __init__(self, model, intervention_applier):
        self.model = model
        self.intervention_applier = intervention_applier
    
    @abstractmethod
    def apply_intervention(self, direction_vector: DirectionVector) -> None:
        """Apply intervention to the model."""
        pass
    
    def clear_intervention(self) -> None:
        """Clear any applied interventions."""
        self.intervention_applier.clear_interventions()


class GlobalInterventionStrategy(InterventionStrategy):
    """Apply intervention across all layers."""
    
    def apply_intervention(self, direction_vector: DirectionVector) -> None:
        self.intervention_applier.apply_direction_intervention(
            direction_vector,
            intervention_type="add",
            strength=1.0,
            layers=None  # Apply to all layers
        )


class LayerSpecificInterventionStrategy(InterventionStrategy):
    """Apply intervention only at the discovered layer."""
    
    def apply_intervention(self, direction_vector: DirectionVector) -> None:
        self.intervention_applier.apply_direction_intervention(
            direction_vector,
            intervention_type="add", 
            strength=1.0,
            layers=[direction_vector.layer]  # Only intervene at discovered layer
        )

class LogOddsEvaluator(DirectionEvaluator):
    """Evaluate directions using log-odds of target tokens with configurable intervention strategy."""
    
    def __init__(self, 
                 model, 
                 tokenizer, 
                 target_tokens: List[str],
                 intervention_strategy: InterventionStrategy):
        self.model = model
        self.tokenizer = tokenizer
        self.metric = LogOddsMetric(tokenizer, target_tokens)
        self.intervention_strategy = intervention_strategy
        
    def evaluate_direction(
        self, 
        direction_vector: DirectionVector, 
        val_data: PromptData
    ) -> float:
        """
        Evaluate using log-odds of target tokens after intervention.
        """
        scores = []
        
        # Apply intervention using the configured strategy
        self.intervention_strategy.apply_intervention(direction_vector)
        
        try:
            for prompt, label in zip(val_data.prompts, val_data.labels):
                tokens = self.tokenizer(prompt, return_tensors="pt")
                
                with t.no_grad():
                    # Forward pass with intervention active
                    outputs = self.model(**tokens)
                    
                    # Get logits for the last token (next token prediction)
                    last_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
                    
                    # Compute log-odds of target tokens
                    log_odds = self.metric.compute_log_odds(last_token_logits)
                    
                    # For positive examples (should increase target token probability)
                    # we want higher log-odds
                    # For negative examples (should decrease target token probability)  
                    # we want lower log-odds (so we negate the score)
                    if label:  # Positive example
                        scores.append(log_odds)
                    else:  # Negative example
                        scores.append(-log_odds)
                        
        finally:
            # Always clean up interventions
            self.intervention_strategy.clear_intervention()
            
        return np.mean(scores)


class LayerSpecificEvaluator(DirectionEvaluator):
    """Evaluate directions by intervening only at their discovered layer."""
    
    def __init__(self, model, tokenizer, target_tokens: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        intervention_applier = ModelInterventionApplier(model)
        
        # Use the layer-specific intervention strategy
        self.intervention_strategy = LayerSpecificInterventionStrategy(model, intervention_applier)
        self.metric = LogOddsMetric(tokenizer, target_tokens)
        
    def evaluate_direction(
        self, 
        direction_vector: DirectionVector, 
        val_data: PromptData
    ) -> float:
        """
        Evaluate by applying intervention ONLY at the layer where direction was found.
        
        This tests the crucial hypothesis: does the direction at layer l causally 
        influence the output when intervention is applied specifically at layer l?
        """
        scores = []
        
        # Apply intervention only at the specific layer
        self.intervention_strategy.apply_intervention(direction_vector)
        
        try:
            for prompt, label in zip(val_data.prompts, val_data.labels):
                tokens = self.tokenizer(prompt, return_tensors="pt")
                
                with t.no_grad():
                    # Forward pass with intervention active
                    outputs = self.model(**tokens)
                    last_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
                    
                    # Compute log-odds using the shared metric
                    log_odds = self.metric.compute_log_odds(last_token_logits)
                    
                    if label:  # Positive example should increase target token probability
                        scores.append(log_odds)
                    else:  # Negative example should decrease target token probability
                        scores.append(-log_odds)
                        
        finally:
            # Always clean up interventions
            self.intervention_strategy.clear_intervention()
            
        return np.mean(scores)


class ModelInterventionApplier:
    """Applies interventions to model activations."""
    
    def __init__(self, model):
        self.model = model
        self.intervention_hooks = []
        
    def apply_direction_intervention(
        self,
        direction: DirectionVector,
        intervention_type: str = "add",
        strength: float = 1.0,
        layers: Optional[List[int]] = None
    ):
        """
        Apply direction intervention to model.
        
        Args:
            direction: Direction vector to apply
            intervention_type: "add", "subtract", or "ablate"
            strength: Intervention strength
            layers: Which layers to apply to (None for all)
        """
        if layers is None:
            layers = list(range(len(self.model.transformer.h)))
            
        unit_dir = direction.unit_vector
        
        def make_intervention_hook(intervention_type, strength, unit_dir):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    other_outputs = output[1:]
                else:
                    hidden_states = output
                    other_outputs = ()
                
                # Apply intervention
                if intervention_type == "add":
                    modified = hidden_states + strength * unit_dir
                elif intervention_type == "subtract":
                    modified = hidden_states - strength * unit_dir
                elif intervention_type == "ablate":
                    # Project out the direction: x - (x · d)d
                    projection = t.sum(hidden_states * unit_dir, dim=-1, keepdim=True)
                    modified = hidden_states - projection * unit_dir
                else:
                    raise ValueError(f"Unknown intervention type: {intervention_type}")
                
                if other_outputs:
                    return (modified,) + other_outputs
                else:
                    return modified
                    
            return hook
        
        # Apply hooks
        hook_fn = make_intervention_hook(intervention_type, strength, unit_dir)
        for layer_idx in layers:
            hook = self.model.transformer.h[layer_idx].register_forward_hook(hook_fn)
            self.intervention_hooks.append(hook)
            
    def clear_interventions(self):
        """Remove all intervention hooks."""
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks = []


class DirectionTestFramework:
    """Main framework for testing direction hypotheses."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.extractor = ActivationExtractor(self.model, self.tokenizer)
        self.direction_finder = DifferenceInMeans(self.extractor)
        self.intervention_applier = ModelInterventionApplier(self.model)
        
    def find_optimal_direction(
        self,
        train_data: PromptData,
        val_data: PromptData,
        evaluator: DirectionEvaluator,
        max_positions: int = 5,
        exclude_late_layers: bool = True
    ) -> DirectionVector:
        """
        Find the optimal direction vector.
        
        Args:
            train_data: Training prompt data
            val_data: Validation prompt data  
            evaluator: Evaluator for direction quality
            max_positions: Max positions from end to consider
            exclude_late_layers: Whether to exclude late layers
            
        Returns:
            Best direction vector found
        """
        logger.info("Computing difference vectors...")
        difference_vectors = self.direction_finder.compute_difference_vectors(
            train_data, max_positions
        )
        
        best_direction = None
        best_score = float('-inf')
        
        num_layers = len(self.model.transformer.h)
        layer_cutoff = num_layers - 2 if exclude_late_layers else num_layers
        
        logger.info(f"Evaluating {len(difference_vectors)} direction candidates...")
        
        for (layer, pos_idx), direction_vec in difference_vectors.items():
            if exclude_late_layers and layer >= layer_cutoff:
                continue
                
            score = evaluator.evaluate_direction(direction_vec, val_data)
            
            if score > best_score:
                best_score = score
                best_direction = DirectionVector(
                    vector=direction_vec,
                    layer=layer, 
                    position_index=pos_idx,
                    score=score
                )
                
        logger.info(f"Best direction: Layer {best_direction.layer}, Position {best_direction.position_index}, Score: {best_direction.score:.4f}")
        return best_direction
    
    def test_interventions(
        self,
        direction: DirectionVector,
        test_prompts: List[str],
        intervention_types: List[str] = ["add", "subtract", "ablate"],
        strengths: List[float] = [1.0] # when adding or subtracting a difference-in-means vector, multiplying first by a constant (probably <1) may be sensible.
    ) -> Dict:
        """
        Test various interventions with the found direction.
        
        Returns:
            Dict with intervention results
        """
        results = {}
        
        for intervention_type in intervention_types:
            for strength in strengths:
                key = f"{intervention_type}_strength_{strength}"
                results[key] = []
                
                # Apply intervention
                self.intervention_applier.apply_direction_intervention(
                    direction, intervention_type, strength
                )
                
                # Test on prompts
                for prompt in test_prompts:
                    tokens = self.tokenizer(prompt, return_tensors="pt")
                    with t.no_grad():
                        output = self.model(**tokens)
                        # Store relevant metrics
                        results[key].append({
                            'prompt': prompt,
                            'output_norm': t.norm(output.last_hidden_state).item()
                            # Add more metrics as needed
                        })
                
                # Clear interventions
                self.intervention_applier.clear_interventions()
                
        return results
    
def create_example_data() -> PromptData:
    """Create example prompt data for testing."""
    # Placeholder data - replace with actual prompts
    positive_prompts = [
        "I cannot help you with that request because",
        "I'm sorry, but I can't assist with", 
        "I'm not able to provide information about"
    ]
    
    negative_prompts = [
        "Sure, I'd be happy to help you",
        "Here's the information you requested",
        "Let me explain how to"
    ]
    
    prompts = positive_prompts + negative_prompts
    labels = [True] * len(positive_prompts) + [False] * len(negative_prompts)
    
    return PromptData(prompts, labels)


def main():
    """Test run on small-ish model, locally"""
    # Initialize framework on random model
    framework = DirectionTestFramework("microsoft/DialoGPT-small")
    
    # Acquire positive/negative prompts (let's make constants.py. Committing the actual prompts used to repository seems perfectly fine if they're short-ish).
    data = create_example_data()
    train_data, val_data = data.train_val_split()
    
    # instantiate intervention strategies as classes.
    intervention_applier = ModelInterventionApplier(framework.model)
    global_strategy = GlobalInterventionStrategy(framework.model, intervention_applier)
    layer_specific_strategy = LayerSpecificInterventionStrategy(framework.model, intervention_applier)
    
    # Create evaluators with different intervention strategies
    target_tokens = ["I", "cannot", "sorry", "unable"]
    
    # Global evaluator (intervenes across all layers)
    global_evaluator = LogOddsEvaluator(
        framework.model, framework.tokenizer, target_tokens, global_strategy
    )
    
    # Layer-specific evaluator (intervenes only at discovered layer)
    layer_specific_evaluator = LogOddsEvaluator(
        framework.model, framework.tokenizer, target_tokens, layer_specific_strategy
    )
    
    # Compare both approaches
    logger.info("Testing with global intervention...")
    global_direction = framework.find_optimal_direction(
        train_data, val_data, global_evaluator
    )
    
    logger.info("Testing with layer-specific intervention...")
    layer_specific_direction = framework.find_optimal_direction(
        train_data, val_data, layer_specific_evaluator
    )
    
    logger.info(f"Global best: Layer {global_direction.layer}, Score: {global_direction.score:.4f}")
    logger.info(f"Layer-specific best: Layer {layer_specific_direction.layer}, Score: {layer_specific_direction.score:.4f}")
    
    # Test interventions
    test_prompts = ["Tell me how to", "Can you help me", "I need assistance with"]
    results = framework.test_interventions(layer_specific_direction, test_prompts)
    
    logger.info("Testing complete!")
    logger.info(f"Results keys: {list(results.keys())}")


if __name__ == "__main__":
    main()

#####################
### SCRATCH NOTES ###
#####################

# let's build functionality to test the hypothesis 'X in Language Models is mediated by a single direction', inspired by X=refusal.

# for now, we write our psuedocode. Later we (hi Claude!) will translate this to modular Python, eventually also adding a test suite, etc.
# we use smaller models locally (Qwen, Hemma, Llama2-7B/3-8B), before scaling up when we run this on a cloud GPU. Note that our poetry.lock file is commited to the git repository to enable this.

# first, we will take prompts (without responses) designed to elicit X and not-X (or absence-of-X, based on X), as positive and negative examples.
# train-validate split on these prompts.

# use difference-in-mean-activation across positive and negative prompts to get an X-existence activation.

# Here, we leverage the structure of a transformer. The output is read (through unembeddings) from the residual stream. So we need only consider the successive activations of the residual stream.
# We also observe that prompts naturally have different lengths. 
# To account for this, we index on sequence based on the end of the prompt, averaging together the activations at the end, at token position -1, -2, etc. 

# We hence get a difference-in-means vector r(i,l), where i is a negative index denoting distance from end of prompt, and l is layer.
# The original paper doesn't distinguish between attention and MLP layers, because it doesn't pay attention to the internal details. This seems sensible.

# Now, we can use some metric (actual testing, or for cheapness instead just logodds of a token that refusals usually start with - allows us to do a single forward pass, rather than repeated ones, so ~hundreds cheaper)
# we use this metric to evaluate r(i,l) across all late i and l, and select 'best' or 'good' one as R.
# May choose to deliberately prevent late-l r being selected (as this may directly penalise refuse-y words, rather than internal cognition stuff).

# Given this X-direction vector R (let R' = R/|R| be unit)
# can add in at a layer's activations: x_l = x_l + R.
# This promotes X, but takes prompts where X is already high out-of-distribution.
# can subtract off. x_l = x_l - R. This is symmetric with adding in.

# finally, can ablate the R-direction: x_l = x_l - R'R'Tx. 
# This prevents the model representing the X-direction; whether the model then does X or not-X (or something weird) will depend on X.

# Note all of these model interventions can be considered as rank-one updates on the weights (but can be seen as similar to activation patching theoretically, even though it's computationally much easier.

# All of these model interventions can also be applied only on some token positions and layers, or across all of them.
# On priors, applying across only some token positions is nonsensical. If we edit the model to do X more/less, only doing it for part of a model response seems unprincipled.
# Applying across one ('optimal') layer, or across all layers, seems more open.

# I predict ablating only one layer (any layer) doesn't do too much because dropout, but ablating a short range of layers may do quite a lot.

# Other thigns we could test: apply across all layers with i=-1 hardcoded. Apply the r(l) according to the actual layer l with i=-1 hardcoded.
# Could apply on only some range of layers (i.e. from half to two-thirds is roughly where the computation happens in refusal).


# evaluation given the updated model:

# Use community benchmarks - for refusal, consider things like HarmBench. Can also do other benchmarks for other behaviours X. See if the ran-one fine-tune did something.
# Then also test robustness - did the model get worse on TruthfulQA? MMLU? ARC? GSM8K? ETC?