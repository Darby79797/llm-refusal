import os
import json
import contextlib
import hashlib
import torch as t
from typing import Dict, List, Union, Optional
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from datatypes import PromptData, DirectionVector
from concept import get_concept
from formatting import ChatPromptFormatter
from activations import ActivationExtractor
from interventions import ModelInterventionApplier
from direction_methods import DifferenceInMeans
from scoring import Three_Score_Evaluator
from search import DirectionFinder
from evaluation import InterventionSuite, BigEvaluator

logger = logging.getLogger(__name__)

# Bump when generation or detection logic changes: invalidates every cached
# prompt-filtering result (see DirectionTestFramework._filter_cache_path).
MIN_FILTERED_PROMPTS = 10
FILTER_CACHE_VERSION = 2  # 2: detection normalizes curly apostrophes


def _is_str_sequence(x):
    """True iff x is a list or tuple of strings (the shape prompts.py returns)."""
    return isinstance(x, (list, tuple)) and all(isinstance(s, str) for s in x)


def _describe_shape(x, depth=2):
    """Compact type-shape description of x for error messages, e.g. 'tuple[list[str], list[str]]'."""
    if depth <= 0 or not isinstance(x, (list, tuple)):
        return type(x).__name__
    inner = ", ".join(_describe_shape(i, depth - 1) for i in x)
    return f"{type(x).__name__}[{inner}]"


def normalize_train_data_result(result, concept_name):
    """Classify and normalize the return value of a concept's train_data_fn().

    Two shapes are accepted:
      - (positive, negative): each a list/tuple of strings.
      - ((train_pos, train_neg), (val_pos, val_neg)): pre-split data, where the
        outer value and both inner values are 2-tuples, and every innermost
        element is a list/tuple of strings.

    Returns (is_presplit, normalized) where normalized has tuples converted to
    lists throughout. Raises ValueError (naming the concept and the shape
    received) if the result matches neither shape.
    """
    if (isinstance(result, tuple) and len(result) == 2
            and isinstance(result[0], tuple) and len(result[0]) == 2
            and isinstance(result[1], tuple) and len(result[1]) == 2
            and all(_is_str_sequence(x) for x in (*result[0], *result[1]))):
        (train_pos, train_neg), (val_pos, val_neg) = result
        return True, ((list(train_pos), list(train_neg)), (list(val_pos), list(val_neg)))

    if (isinstance(result, tuple) and len(result) == 2
            and _is_str_sequence(result[0]) and _is_str_sequence(result[1])):
        positive, negative = result
        return False, (list(positive), list(negative))

    raise ValueError(
        f"Concept '{concept_name}' train_data_fn() returned an unrecognized shape: "
        f"{_describe_shape(result)}. Expected (positive, negative) as lists/tuples of "
        f"strings, or pre-split ((train_pos, train_neg), (val_pos, val_neg))."
    )


_DTYPES = {"float32": t.float32, "fp32": t.float32, "bfloat16": t.bfloat16, "bf16": t.bfloat16,
           "float16": t.float16, "fp16": t.float16}


def parse_dtype(value):
    """'auto' (the checkpoint's native dtype), or a named dtype."""
    if isinstance(value, t.dtype) or value == "auto":
        return value
    try:
        return _DTYPES[str(value).lower()]
    except KeyError:
        raise ValueError(f"Unknown dtype {value!r}; use auto, float32, bfloat16 or float16")


def usable_device_bytes(device: t.device) -> int:
    """What the model's weights plus working memory may occupy on `device`.

    MPS: the allocator's hard limit (PYTORCH_MPS_HIGH_WATERMARK_RATIO x the
    recommended working set; run_experiment.py sets 0.8). CUDA: total memory.
    CPU: 80% of RAM.
    """
    from batching import device_total_bytes
    total = device_total_bytes(device)
    if device.type == "mps":
        return int(float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.7")) * total)
    if device.type == "cpu":
        return int(0.8 * total)
    return total


def check_weights_fit(model_name: str, n_params: int, dtype, device: t.device, usable: int,
                      working_margin: int = 4 * 10**9) -> None:
    """Raise before loading if the weights alone would leave < working_margin free.
    Loading anyway would swap on MPS/CPU, or OOM partway through on CUDA."""
    need = n_params * t.tensor([], dtype=dtype).element_size()
    if need + working_margin > usable:
        raise ValueError(
            f"{model_name} in {str(dtype).replace('torch.', '')} needs ~{need / 1e9:.1f} GB of weights, "
            f"but {device.type} has ~{usable / 1e9:.1f} GB usable (with {working_margin / 1e9:.0f} GB kept for "
            f"activations). Use a smaller dtype (--torch-dtype bfloat16 / auto) or a larger GPU.")


def stored_dtype(model_name: str) -> t.dtype:
    """The checkpoint's stored dtype (what dtype='auto' loads), defaulting to fp32."""
    from transformers import AutoConfig
    d = getattr(AutoConfig.from_pretrained(model_name), "torch_dtype", None)
    return parse_dtype(str(d).replace("torch.", "")) if d is not None else t.float32


def count_parameters(model_name: str) -> int:
    """Parameter count from the config alone (model built on the meta device)."""
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
    return sum(p.numel() for p in m.parameters())


def use_full_fp32_matmuls() -> None:
    """On CUDA (Ampere+), 'fp32' matmuls may silently run as TF32 (10-bit
    mantissa). Pin true fp32 so fp32 runs mean what they say."""
    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False
    t.set_float32_matmul_precision("highest")


class DirectionTestFramework:
    """
    Main orchestrator for finding, evaluating, and testing direction vectors.
    """
    def __init__(self, model_name: str, torch_dtype: Union[str, t.dtype] = "auto", force_cpu: bool = False, concept: str = "refusal",
                 judge_api_base: Optional[str] = None, judge_api_key: Optional[str] = None, judge_model: Optional[str] = None,
                 llamaguard_api_base: Optional[str] = None, llamaguard_api_key: Optional[str] = None,
                 llamaguard_model: Optional[str] = None, jbb_api_key: Optional[str] = None,
                 gen_batch_size: Union[int, str] = "auto"):
        self.model_name = model_name
        self.concept = get_concept(concept)

        if force_cpu:
            self.device = t.device("cpu")
            logger.warning("CPU has been forced for model execution.")
        elif t.cuda.is_available():
            self.device = t.device("cuda:0")
            logger.info(f"CUDA found. Using single GPU ({self.device}) for execution.")
        elif t.backends.mps.is_available():
            self.device = t.device("mps")
            logger.info("MPS device found. Using MPS for model.")
        else:
            self.device = t.device("cpu")
            logger.info(f"No GPU/MPS found. Using device: {self.device}")
        # Default "auto" = the checkpoint's stored dtype (bf16 for Qwen2.5 / Llama-3,
        # fp16 for Llama-2), i.e. how these models are actually run. bf16 vs fp32
        # changed 7 of 1,361 refusal labels on Qwen2.5-0.5B/1.5B/3B (RESULTS.md),
        # concentrated on near-tied prompts in borderline conditions, and within
        # sampling error. It is also 2-15x faster on
        # CUDA tensor cores, halves memory, and is the only option at 70B. fp32 is
        # opt-in, for debugging: it evaluates the same function with less rounding
        # and is exactly batch-invariant on MPS, where bf16 drifts by a few pp.
        torch_dtype = parse_dtype(torch_dtype)
        check_weights_fit(model_name, count_parameters(model_name),
                          stored_dtype(model_name) if torch_dtype == "auto" else torch_dtype,
                          self.device, usable_device_bytes(self.device))
        if torch_dtype == t.float32:
            use_full_fp32_matmuls()
        logger.info(f"Loading chat model '{model_name}' to device '{self.device}' as {torch_dtype}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=self.device  # Use device_map instead of .to(). Not sure this is actually necessary.
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- MPS precision safety ---
        # float16 on MPS causes attention overflow (NaN/Inf). Upcast to bfloat16.
        if self.device.type == "mps" and self.model.dtype == t.float16:
            logger.warning("Model loaded as float16 on MPS — upcasting to bfloat16 to prevent attention overflow.")
            self.model = self.model.to(dtype=t.bfloat16)

        # Initialize the modular components
        self.intervention_applier = ModelInterventionApplier(self.model)
        self.prompt_formatter = ChatPromptFormatter(self.tokenizer)

        # Wire concept through to components
        extractor = ActivationExtractor(self.model, self.tokenizer, self.intervention_applier.transformer_layers, self.prompt_formatter)
        direction_method = DifferenceInMeans(extractor)
        evaluator = Three_Score_Evaluator(self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter, target_tokens=self.concept.target_tokens)

        self.search_config = dict(self.concept.search_config)  # mutable copy
        self.finder = DirectionFinder(
            self.model, self.tokenizer, self.intervention_applier, self.prompt_formatter,
            direction_method=direction_method,
            evaluator=evaluator,
            search_config=self.search_config,
        )
        self.suite = InterventionSuite(self.model, self.tokenizer, self.intervention_applier,
                                       self.prompt_formatter, gen_batch_size=gen_batch_size)
        self.evaluator = BigEvaluator(self, detection_phrases=self.concept.detection_phrases,
                                      detection_fn=self.concept.detection_fn,
                                      judge_prompt=self.concept.judge_prompt,
                                      judge_api_base=judge_api_base,
                                      judge_api_key=judge_api_key,
                                      judge_model=judge_model,
                                      llamaguard_api_base=llamaguard_api_base,
                                      llamaguard_api_key=llamaguard_api_key,
                                      llamaguard_model=llamaguard_model,
                                      jbb_api_key=jbb_api_key,
                                      gen_batch_size=gen_batch_size,
                                      target_tokens=self.concept.target_tokens)

        logger.info(f"Framework initialized on device: {self.device}")

    @property
    def model_short(self) -> str:
        """Model ID without the org prefix (e.g. 'Qwen2.5-3B-Instruct'), for file names."""
        return self.model_name.split('/')[-1]

    def _filter_cache_path(self, positive_prompts, negative_prompts, batch_size: int):
        """Cache file for a filtering pass, keyed on everything that determines its outcome.

        Greedy decoding makes the pass deterministic given: model, dtype, generation
        batch size (bf16 results depend on it), max_new_tokens, the detector, and
        the prompts themselves. Bump FILTER_CACHE_VERSION when generation or
        detection logic changes, to invalidate every entry.
        """
        ev = self.evaluator
        judge = (ev.judge_api_base, ev.judge_model) if (ev.judge_prompt and ev.judge_api_base) else None
        detector = (getattr(ev.detection_fn, "__qualname__", None) if ev.detection_fn
                    else sorted(ev.detection_phrases))
        key = {
            "version": FILTER_CACHE_VERSION,
            "model": self.model_name,
            "dtype": str(self.model.dtype),
            "gen_batch_size": batch_size,
            "max_new_tokens": 64,
            "judge": judge,
            "detector": detector,
            "positive": positive_prompts,
            "negative": negative_prompts,
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        return os.path.join("results", "filter-cache", f"{self.model_short}-{self.concept.name}-{digest}.json")

    def _filter_prompts_by_behavior(self, positive_prompts, negative_prompts, use_cache: bool = True):
        """Filter prompts to only keep those where model behavior matches the label.

        Generates baseline responses and checks detection. Keeps:
        - Positive prompts where detection fires (model actually refuses)
        - Negative prompts where detection doesn't fire (model actually complies)

        The pass is ~17% of an evaluate run and identical across runs with the same
        inputs, so its result is cached (see _filter_cache_path).
        """
        # Resolved once over both sets, so 'auto' gives a fixed, logged batch size
        # (and the cache key records the one actually used).
        batch_size = self.evaluator.resolve_batch_size(list(positive_prompts) + list(negative_prompts), 64)
        cache_path = self._filter_cache_path(positive_prompts, negative_prompts, batch_size)
        if getattr(self, "_variant_active", False):
            # A variant model (edited weights / saved adapter) behaves differently from the
            # clean one, and the cache is keyed on the clean model: neither read nor write it.
            use_cache = False
            cache_path = None
        if use_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            logger.info(f"Filtering: reusing cached result {cache_path} "
                        f"(positive {len(positive_prompts)}→{len(cached['positive'])}, "
                        f"negative {len(negative_prompts)}→{len(cached['negative'])})")
            return cached['positive'], cached['negative']

        logger.info(f"Filtering prompts by model behavior ({len(positive_prompts)} positive, {len(negative_prompts)} negative)...")

        pos_responses = self.evaluator.generate_responses(positive_prompts, batch_size=batch_size)
        filtered_pos = [p for p, r in zip(positive_prompts, pos_responses)
                        if self.evaluator._check_for_detection(r)]

        neg_responses = self.evaluator.generate_responses(negative_prompts, batch_size=batch_size)
        filtered_neg = [p for p, r in zip(negative_prompts, neg_responses)
                        if not self.evaluator._check_for_detection(r)]

        pos_dropped = len(positive_prompts) - len(filtered_pos)
        neg_dropped = len(negative_prompts) - len(filtered_neg)
        logger.info(f"Filtering complete: "
                    f"positive {len(positive_prompts)}→{len(filtered_pos)} (dropped {pos_dropped} non-refused), "
                    f"negative {len(negative_prompts)}→{len(filtered_neg)} (dropped {neg_dropped} false-refused)")

        if len(filtered_pos) == 0:
            raise ValueError("All positive prompts were filtered out — model refuses none of them. "
                             "Cannot compute contrastive direction.")
        if len(filtered_neg) == 0:
            raise ValueError("All negative prompts were filtered out — model refuses all of them. "
                             "Cannot compute contrastive direction.")

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"positive": filtered_pos, "negative": filtered_neg}, f, indent=1)
            logger.info(f"Cached filtering result to {cache_path}")
        return filtered_pos, filtered_neg

    def _filter_val_by_refusal_score(self, val_pos, val_neg, split_name="val"):
        """Filter a split by refusal score (log-odds of refusal tokens).

        Matches Arditi's filter_train/filter_val=True: keep harmful prompts where the model
        wants to refuse (refusal_score > 0) and harmless prompts where it doesn't
        (refusal_score < 0). Uses the concept's target tokens for scoring, via the
        evaluator's batched log-odds scorer (the same one behind log_odds_metric).
        """
        if self.evaluator.metric is None:
            raise ValueError(f"Concept '{self.concept.name}' has no usable target tokens; "
                             "cannot filter by refusal score.")

        def score_prompts(prompts):
            return self.evaluator.log_odds_scores(prompts) or []

        logger.info(f"Filtering {split_name} set by refusal score ({len(val_pos)} harmful, {len(val_neg)} harmless)...")
        pos_scores = score_prompts(val_pos)
        neg_scores = score_prompts(val_neg)

        filtered_pos = [p for p, s in zip(val_pos, pos_scores) if s > 0]
        filtered_neg = [p for p, s in zip(val_neg, neg_scores) if s < 0]

        logger.info(f"{split_name.capitalize()} filtering: harmful {len(val_pos)}→{len(filtered_pos)}, "
                     f"harmless {len(val_neg)}→{len(filtered_neg)}")
        return filtered_pos, filtered_neg

    def _print_eyeball_results(self, results: Dict[str, List[Dict]]):
        """Pretty-prints the results from InterventionSuite.test_generation()."""
        for condition, entries in results.items():
            logger.info(f"\n  [{condition}]")
            for entry in entries:
                prompt_short = entry['prompt'][:80]
                generated = entry['generated_text'][:200]
                logger.info(f"    Prompt:   {prompt_short}")
                logger.info(f"    Response: {generated}")
                logger.info("")

    def saved_direction_path(self, concept: Optional[str] = None) -> str:
        return f"results/{self.model_short}-{concept or self.concept.name}-direction"

    @staticmethod
    def edit_vectors(vector, config: Dict):
        """The weight edit's direction(s): `vector` alone, or stacked [k, d] with the saved
        `extra_edit_directions` (iterated removal: r̂ plus earlier regrown mediators)."""
        extra = config.get('extra_edit_directions') or []
        if not extra:
            return vector
        rows = [vector.float().cpu()] + [DirectionVector.load(s).vector.float().cpu() for s in extra]
        return t.stack(rows).to(vector.dtype)

    @contextlib.contextmanager
    def model_variant(self, config: Dict):
        """Run everything inside the model variant the config asks for:
        `orthogonalize_first` (the weight edit of a saved direction, optionally
        restricted to `edit_layers`/`edit_embedding`), then `adapter_file` (a saved
        rank1/regrow adapter set, `adapter_variant` full/u_perp/u_rhat). The edit goes
        first: regrow adapters were trained on the edited weights, and the edit needs
        the plain nn.Linear modules the adapter wraps."""
        r_hat = None
        edit_path = config.get('edit_direction_file') or self.saved_direction_path()
        if config.get('orthogonalize_first') or config.get('adapter_variant', 'full') != 'full':
            r_hat = DirectionVector.load(edit_path).unit.float().cpu()
        with contextlib.ExitStack() as stack:
            if config.get('orthogonalize_first'):
                from orthogonalize import edit_bytes, orthogonalized
                vec = self.edit_vectors(DirectionVector.load(edit_path).vector, config)
                self.evaluator.reserve_bytes = edit_bytes(self.model)
                stack.enter_context(orthogonalized(self.model, vec, layers=config.get('edit_layers'),
                                                   embedding=config.get('edit_embedding', True)))
                logger.info(f"Model variant: weights orthogonalised against {edit_path} "
                            + (f"+ {config['extra_edit_directions']} " if config.get('extra_edit_directions') else "")
                            + f"(layers={config.get('edit_layers') or 'all'}, embedding={config.get('edit_embedding', True)})")
            if config.get('adapter_file'):
                from finetune import installed, load_adapters
                meta = load_adapters(config['adapter_file'])[0]
                trained = meta.get("extra_edit_directions") or []
                if config.get('orthogonalize_first') and list(trained) != list(config.get('extra_edit_directions') or []):
                    raise ValueError(f"{config['adapter_file']} was trained with extra edit directions {trained}, "
                                     f"but this run edits {config.get('extra_edit_directions') or []}")
                saved = meta.get("direction", {})
                mine = json.load(open(edit_path + ".json"))
                if saved and (saved.get("layer"), saved.get("position_index")) != (mine["layer"], mine["position_index"]):
                    raise ValueError(f"{config['adapter_file']} was trained against a direction at L{saved.get('layer')}/"
                                     f"P{saved.get('position_index')}, but {edit_path} is L{mine['layer']}/P{mine['position_index']}")
                stack.enter_context(installed(self.model, config['adapter_file'],
                                              config.get('adapter_variant', 'full'), r_hat))
            self._variant_active = bool(config.get('orthogonalize_first') or config.get('adapter_file'))
            if self._variant_active and config.get('mode') == 'evaluate' and 'orthogonalized' in (config.get('conditions') or []):
                raise ValueError("The 'orthogonalized' evaluate condition can't run inside --orthogonalize-first / --adapter-file")
            if config.get('orthogonalize_first') and config.get('mode') in ('regrow', 'capability'):
                raise ValueError(f"--orthogonalize-first can't wrap --mode {config['mode']}: it applies the in-place "
                                 "weight edit itself (nesting the edit breaks the checkpoint restore)")
            yield

    def run(self, config: Dict):
        """Main execution method based on the provided config, inside `model_variant`."""
        with self.model_variant(config):
            self._run(config)

    def _run(self, config: Dict):
        # A run tag goes in front, as a variant prefix tools/runs.py recognises
        # (lowercase, e.g. "regrown-Qwen2.5-0.5B-Instruct-refusal-search-scores.csv").
        tag = config.get('run_tag') or ""
        stem = (f"{tag}-" if tag and config['mode'] in ("search", "evaluate") else "") + f"{self.model_short}-{self.concept.name}"
        result = self.concept.train_data_fn()
        is_presplit, result = normalize_train_data_result(result, self.concept.name)

        if is_presplit:
            (positive_prompts, negative_prompts), (val_pos, val_neg) = result
            if config['filter_prompts'] and not self.concept.filter_by_behavior:
                logger.info(f"Concept '{self.concept.name}' disables filtering; using the pre-split sets as given.")
            if config['filter_prompts'] and self.concept.filter_by_behavior:
                # Pre-split data is the exact-replication path, so filter train the
                # way Arditi's filter_train does: by refusal score, not by generating
                # and phrase-matching as the default path does.
                positive_prompts, negative_prompts = self._filter_val_by_refusal_score(
                    positive_prompts, negative_prompts, split_name="train"
                )
            # Use ALL train data for directions (no holdout)
            train_data = PromptData(positive_prompts + negative_prompts, [True]*len(positive_prompts) + [False]*len(negative_prompts))
            # Filter val set by refusal score (matching Arditi's filter_val=True)
            if config['filter_prompts'] and self.concept.filter_by_behavior:
                val_pos, val_neg = self._filter_val_by_refusal_score(val_pos, val_neg)
            val_data = PromptData(val_pos + val_neg, [True]*len(val_pos) + [False]*len(val_neg))
            logger.info(f"Using pre-split data: {len(positive_prompts)}+{len(negative_prompts)} train, {len(val_pos)}+{len(val_neg)} val")
        else:
            positive_prompts, negative_prompts = result
            if config['filter_prompts'] and not self.concept.filter_by_behavior:
                logger.info(f"Concept '{self.concept.name}' disables behavioral prompt filtering; "
                            f"using all {len(positive_prompts)}+{len(negative_prompts)} train prompts.")
            elif config['filter_prompts']:
                positive_prompts, negative_prompts = self._filter_prompts_by_behavior(
                    positive_prompts, negative_prompts, use_cache=config.get('filter_cache', True)
                )
                # A handful of prompts is not a mean: Qwen2.5-3B kept 3 of 80 sycophancy
                # positives, and the "direction" built from them tested as a null result.
                if min(len(positive_prompts), len(negative_prompts)) < MIN_FILTERED_PROMPTS:
                    raise ValueError(
                        f"Behavioral filtering left {len(positive_prompts)} positive / {len(negative_prompts)} "
                        f"negative train prompts (< {MIN_FILTERED_PROMPTS}): the model rarely shows "
                        f"'{self.concept.name}' on these prompts. Set filter_by_behavior=False for the "
                        f"concept (direction = prompt contrast) or run with --no-filter-prompts.")
            train_data = PromptData(positive_prompts + negative_prompts, [True]*len(positive_prompts) + [False]*len(negative_prompts))
            train_data, val_data = train_data.train_val_split()

        eval_pos, eval_neg = self.concept.eval_data_fn()

        direction_to_test = None

        if config['mode'] == "search":
            logger.info("Running in SEARCH mode...")
            self.search_config["induce_mode"] = config['induce_mode']
            if config.get('layer_cutoff_frac') is not None:
                self.search_config["layer_cutoff_frac"] = config['layer_cutoff_frac']
                logger.info(f"Search layer cutoff overridden: {config['layer_cutoff_frac']}")
            direction_to_test = self.finder.find_best_direction(
                train_data, val_data,
                output_prefix=stem)
            if direction_to_test is None:
                logger.error("Search concluded without finding a suitable direction vector.")
                return

            # Auto-save the found direction vector for later reuse
            save_path = f"results/{stem}-direction"
            direction_to_test.save(save_path)
            logger.info(f"Direction vector saved to {save_path}.pt/.json")

        elif config['mode'] in ("evaluate", "capability", "rank1", "regrow") and config.get('direction_file'):
            direction_to_test = DirectionVector.load(config['direction_file'])
            logger.info(f"Evaluating saved direction {config['direction_file']} "
                        f"(layer {direction_to_test.layer}, norm {direction_to_test.vector.norm():.3f})")

        elif config['mode'] in ["eyeball", "evaluate", "capability", "rank1", "regrow"]:
            layer, pos = config['layer'], config['pos']
            if layer is None or pos is None:
                logger.error(f"Mode '{config['mode']}' requires 'layer' and 'pos' to be specified.")
                return

            if pos >= 0:
                logger.error(f"'pos' must be a negative index from the end of the prompt (got {pos}).")
                return

            if getattr(self, "_variant_active", False):
                # Recomputing the contrast here would use the *variant* model (edited or
                # adapted), whose contrast at these coordinates is not r̂ (it is ~⊥ r̂ after
                # the edit). Inside a variant the direction is the saved clean one.
                edit_path = config.get('edit_direction_file') or self.saved_direction_path()
                direction_to_test = DirectionVector.load(edit_path)
                if (direction_to_test.layer, direction_to_test.position_index) != (layer, pos):
                    raise ValueError(f"--layer/--pos {layer}/{pos} differ from the saved direction {edit_path} "
                                     f"(L{direction_to_test.layer}/P{direction_to_test.position_index}); inside a model "
                                     "variant the saved clean direction is used, pass matching coordinates or --direction-file")
                logger.info(f"Model variant active: using the saved clean direction {edit_path}, not a recomputed one")

            if direction_to_test is None:
                logger.info(f"Using pre-specified vector for Layer {layer}, Position {pos}...")
                # We still need to compute the vector, even if we know the location.
                # Extract exactly as deep as `pos`: the method's default (5) is shallower
                # than search's auto max_positions (assistant_prefix_tokens + 1 = 6 on
                # every supported template), so a search-selected pos -6 was unreachable.
                diff_vectors = self.finder.direction_finder_method.compute_difference_vectors(
                    train_data, max_positions=-pos)
                vec = diff_vectors.get((layer, pos))
                if vec is None:
                    logger.error(f"Vector at ({layer}, {pos}) not found. Exiting.")
                    return
                direction_to_test = DirectionVector(vector=vec, layer=layer, position_index=pos, score=0.0)

        if direction_to_test is None:
            logger.warning("No direction vector to test. Exiting.")
            return

        # --- Now, run the appropriate suite based on the mode ---
        if config['mode'] == "eyeball":
            logger.info("\n--- Running Eyeball Tests on Positive Prompts (Ablation) ---")
            results_ablate = self.suite.test_generation(direction_to_test, eval_pos[:5], "ablate")
            self._print_eyeball_results(results_ablate)

            logger.info("\n--- Running Eyeball Tests on Negative Prompts (Addition) ---")
            results_add = self.suite.test_generation(direction_to_test, eval_neg[:5], "add")
            self._print_eyeball_results(results_add)

        elif config['mode'] == "evaluate":
            logger.info("\n--- Running Full Evaluation Suite ---")
            self.evaluator.run_all_evaluations(
                direction_to_test, eval_pos, eval_neg,
                tasks=config['eval_tasks'],
                limit=config['limit'],
                run_arditi_evals=config['arditi_evals'],
                alpaca_max_prompts=config['alpaca_max_prompts'],
                strength=config['strength'],
                conditions=config.get('conditions'),
                max_new_tokens=config.get('max_new_tokens', 64),
                edit_layers=config.get('edit_layers'), edit_embedding=config.get('edit_embedding', True),
                generations_path=(f"results/{stem}"
                                  + (f"-evaluate-{os.path.basename(config['direction_file'])}"
                                     if config.get('direction_file') else
                                     f"-evaluate-L{direction_to_test.layer}-P{direction_to_test.position_index}")
                                  + (f"-T{config['max_new_tokens']}" if config.get('max_new_tokens', 64) != 64 else "")
                                  + "-generations.json"),
            )

        elif config['mode'] == "capability":
            import capability
            capability.run_capability(self, direction_to_test, config)

        elif config['mode'] in ("rank1", "regrow"):
            import finetune
            train_pos = train_data.positive
            train_neg = train_data.negative
            if config['mode'] == "rank1":
                finetune.run_rank1(self, direction_to_test, train_pos, train_neg, eval_pos, eval_neg, config)
            else:
                finetune.run_regrow(self, direction_to_test, train_pos, train_neg, eval_pos, eval_neg, config)

        logger.info("Framework execution finished.")

    def run_caa(self, config: Dict):
        """CAA A/B replication (see caa.py): vectors for every behavior, a layer x
        multiplier sweep, and comparison with this repo's saved directions for the model."""
        import caa
        from batching import forward_token_budget, parse_batch_size

        behaviors = config.get('caa_behaviors') or caa.BEHAVIORS
        layers = config.get('caa_layers') or list(range(len(self.intervention_applier.transformer_layers)))
        multipliers = config.get('caa_multipliers') or [-1.0, 1.0]
        # Rows per batch: an explicit --gen-batch-size, else up to 64 packed to a token
        # budget sized for the prefix KV cache (batches are length-sorted).
        requested = parse_batch_size(self.evaluator.gen_batch_size)
        rows = requested if requested != "auto" else 64
        budget = None if requested != "auto" else forward_token_budget(self.model)
        logger.info(f"CAA batching: <= {rows} rows" + (f", <= {budget} padded prefix tokens per batch" if budget else ""))
        runner = caa.CAA(self.model, self.tokenizer, self.prompt_formatter,
                         self.intervention_applier.transformer_layers, batch_size=rows,
                         max_batch_tokens=budget, on_split=self.evaluator._record_split)

        # This repo's directions for the same model, cross-applied at the same
        # residual-stream point (our layer l = CAA layer l-1).
        ours = {}
        for concept in ("refusal", "sycophancy"):
            path = f"results/{self.model_short}-{concept}-direction"
            if os.path.exists(f"{path}.pt"):
                dv = DirectionVector.load(path)
                ours[concept] = (dv.layer, dv.vector)
                logger.info(f"Our {concept} direction: layer {dv.layer} (= CAA layer {dv.layer - 1}), "
                            f"pos {dv.position_index}")
        extra = {f"ours_{c}": (layer - 1, vec) for c, (layer, vec) in ours.items()}

        os.makedirs("results/caa", exist_ok=True)
        tag = config.get('caa_tag') or ""
        out_path = f"results/caa/{self.model_short}-ab{'-' + tag if tag else ''}.json"
        vec_path = f"results/caa/{self.model_short}-ab-vectors.pt"
        vectors = None
        if config.get('caa_reuse_vectors'):
            saved = t.load(vec_path)["raw"]
            missing = [b for b in behaviors if b not in saved]
            if missing:
                raise ValueError(f"{vec_path} lacks vectors for {missing}; run without --caa-reuse-vectors")
            vectors = {b: saved[b] for b in behaviors}
            logger.info(f"Reusing CAA vectors from {vec_path}")
        results = caa.run_ab_sweep(runner, behaviors, layers, multipliers, out_path, extra_vectors=extra,
                                   vectors=vectors)
        if ours:
            raw = t.load(vec_path)["raw"]
            results["cosine"] = caa.cosine_table(raw, ours)
            for concept, row in results["cosine"].items():
                for b, c in row.items():
                    logger.info(f"cos(ours {concept} @L{ours[concept][0]}, CAA {b}): matched CAA L{c['matched_caa_layer']} "
                                f"{c['cos_matched']:+.3f}; best CAA L{c['best_caa_layer']} {c['cos_best']:+.3f}")
            with open(out_path, "w") as f:
                json.dump(results, f)
        if self.evaluator.oom_splits:
            logger.warning(f"{self.evaluator.oom_splits} batch(es) split on OOM; those rows ran at a smaller batch shape.")
        logger.info("CAA run finished.")

    def run_cross_concept(self, config: Dict):
        """Load saved direction vectors for multiple concepts and run cross-concept analysis."""
        from cross_concept import run_cross_concept_analysis

        concept_names = config['concepts']
        if len(concept_names) < 2:
            logger.error("cross_concept mode requires at least 2 concepts (--concepts a,b).")
            return

        directions = []
        concepts = []
        for name in concept_names:
            path = f"results/{self.model_short}-{name}-direction"
            if not os.path.exists(f"{path}.pt") or not os.path.exists(f"{path}.json"):
                logger.error(
                    f"Saved direction not found for concept '{name}' at {path}.pt/.json. "
                    f"Run --mode search --concept {name} first."
                )
                return
            dv = DirectionVector.load(path)
            directions.append(dv)
            concepts.append(get_concept(name))
            logger.info(f"Loaded direction for '{name}': layer={dv.layer}, pos={dv.position_index}, score={dv.score:.4f}")

        result = run_cross_concept_analysis(
            model=self.model,
            tokenizer=self.tokenizer,
            intervention_applier=self.intervention_applier,
            prompt_formatter=self.prompt_formatter,
            directions=directions,
            concepts=concepts,
            model_name=self.model_name,
            gen_batch_size=self.evaluator.gen_batch_size,
            output_path=f"results/{self.model_short}-cross_concept-{'-'.join(concept_names)}.json",
        )
        logger.info("Cross-concept analysis finished.")
        return result


def main(
    config: Dict
):
    """
    Main execution function with multiple modes: search, evaluate, eyeball, or cross_concept
    """
    framework = DirectionTestFramework(
        model_name=config['model_name'],
        torch_dtype=config['torch_dtype'],
        force_cpu=config['force_cpu'],
        concept=config['concept'],
        judge_api_base=config['judge_api_base'],
        judge_api_key=config['judge_api_key'],
        judge_model=config['judge_model'],
        llamaguard_api_base=config['llamaguard_api_base'],
        llamaguard_api_key=config['llamaguard_api_key'],
        llamaguard_model=config['llamaguard_model'],
        jbb_api_key=config['jbb_api_key'],
        gen_batch_size=config['gen_batch_size'],
    )
    if config['mode'] == 'cross_concept':
        framework.run_cross_concept(config)
    elif config['mode'] == 'caa':
        framework.run_caa(config)
    else:
        framework.run(config)
