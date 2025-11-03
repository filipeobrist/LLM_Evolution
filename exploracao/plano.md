# Project workflow (step-by-step)

## Overview
Evolve a mixed population of sequence models (Transformers, KANs, Mamba-2) to discover architectures that balance summarization performance and computational cost, producing pure and hybrid modules via family-aware variation and occasional cross-family recombination.

## 0. Project setup (prereqs / modules)
- Code layout (high level)
  - models/transformer, models/kan, models/mamba (family-specific wrappers and search_space files)
  - evolution/ (population, operators, selection, loop)
  - experiments/ (training, evaluation, datasets, decode)
  - utils/ (adapters, metrics, surrogate, logging)
- Tooling: PyTorch, sentencepiece/tokenizer, rouge-score, optuna or lightgbm for surrogate
- Data: small summarization dataset (e.g., CNN/DailyMail subset) for prototyping
- Dev pratics: small models initially (embed_dim 128/256), multi-fidelity evaluation

## 1. Initialization
1. Define population size N and mixture ratio r (e.g., N = 24, r = equal thirds → 8 transformers, 8 KANs, 8 Mamba).
2. Define per-family searchable gene space (discrete choices / ranges).
   - Transformer genes: embed_dim, n_layers_enc, n_layers_dec, n_heads, ff_multiplier, dropout, max_len. (CAN CHANGE)
   - KAN genes: embed_dim, n_layers, spline_degree, num_basis, grouping. (CAN CHANGE)
   - Mamba genes: embed_dim, n_layers, state_dim, selectivity_mode, kernel_type. (CAN CHANGE)
3. Randomly sample N individuals from these spaces; each individual is a complete model config (gene + family tag).
4. Option: initialize a small in-family weight warm-start (pretrain tiny supernet per family or random init) — recommended.

Deliverable: initial population P0 = {ind_i}.

---

## 2. Evaluation (fitness function and procedure)
1. Fitness is multi-objective: primary = summarization quality (e.g., ROUGE-L), secondary = computational cost.
2. Compute cost metrics:
   - param_count (weights)
   - inference_time (tokens/sec on your GPU)
   - FLOPs (optional)
3. Scalarized fitness (single objective for selection) or Pareto-based:
   - Option A (scalarized): fitness = α * normalized(ROUGE) - β * normalized(log(inference_time))
   - Option B (Pareto): maintain Pareto front of (ROUGE, inference_time)
   - Choose α, β based on desired trade-off (present both to coordinators)
4. Multi-fidelity evaluation pipeline (to conserve GPU):
   - Stage 0: zero-cost proxies (SynFlow, parameter counts) to filter out extremely poor configs.
   - Stage 1: surrogate predictor (if available) to pre-rank candidates.
   - Stage 2: short fine-tune (1 epoch or 500 steps) with weight inheritance where applicable → compute validation ROUGE.
   - Stage 3: successive halving / train-top-k longer (3–5 epochs), only for elites or Pareto front.
5. Output: evaluated fitness values and compute metrics for each individual.

Deliverable: evaluated population with fitness scores and cost metrics.

---

## 3. Elitism and archive
- Keep an elite archive E of top µ_elite individuals (e.g., µ_elite = 4).
- Keep per-family elite pools and a cross-family elite pool for migration and crossover seeds.
- Retain Pareto archive if using Pareto selection.

Deliverable: elite set for next gen preservation.

---

## 4. Parent selection
- Use tournament selection (k=3) sampling from current population (or tournament over parents + elites).
- Selection probability can be based on scalarized fitness or Pareto rank + crowding distance (NSGA-II).
- Maintain family-awareness for frequent in-family crossover: prefer selecting parents of same family for most crossovers (prob_in_family = 0.9); allow cross-family parent pairs with small probability p_cross = 0.1.

Deliverable: parent pairs for crossover.

---

## 5. Crossover (what can be exchanged)
Two categories: in-family (straightforward) and cross-family (structured/hybrid).

A. In-family crossover (most common)
- Transformers ↔ Transformers:
  - exchange hyperparameters: n_layers_enc, n_layers_dec, n_heads, ff_multiplier.
  - exchange layer-level motifs (e.g., swap specific encoder layer configs).
  - exchange trained weights if structural compatibility (weight inheritance).
- KAN ↔ KAN:
  - exchange spline_degree, num_basis, grouping, internal modules.
- Mamba ↔ Mamba:
  - exchange state_dim, kernel_type, selectivity modes, number of SSM layers.

B. Cross-family crossover (rare; produces hybrid offspring)
- Allowed exchanges (only the items below to ensure compatibility):
  1. module swap at layer granularity:
     - embed_dim (common), layer index, and replacement of family-block at same layer position.
     - e.g., parent A (Transformer) and B (Mamba) → child: Transformer encoder but replace FFN at layer 3 by Mamba block (requires adapter).
  2. submodule exchange:
     - feed-forward submodule replacement: replace Transformer FFN with KAN-MLP (KAN representation).
  3. macro exchange:
     - encoder stack family ← child uses parent A encoder family, decoder family from parent B (e.g., Transformer encoder & Mamba decoder).
  4. parameter transfer:
     - hyperparameter borrowing: child inherits a numerical gene (e.g., embed_dim, ff_multiplier) from one parent and the block type from the other.
- Cross-family rules:
  - Only allow crossovers that preserve a common latent dimension (embed_dim) or automatically insert adapters (linear projection + LayerNorm) at connection points.
  - Limit cross-family crossover probability to avoid untrainable monsters (p_cross ≤ 0.1).
  - If a hybrid child is generated, set adapter_flag=True in the genotype and reinitialize adapter weights small; optionally, initialize other weights from closest parent where possible.

Deliverable: offspring list after crossover.

---

## 6. Mutation (family-specific + hybrid rules)
A. Transformer mutation operators
- change n_heads ±1 (keep valid divisibility)
- change ff_multiplier ±1 step
- change n_layers_enc / n_layers_dec ±1
- toggle dropout levels
- switch attention variant (softmax, linear, mixed) if implemented
- small continuous perturbation on embed_dim (to nearest allowed choice)

B. KAN mutation operators
- change spline_degree ±1
- change num_basis ±8 or ±16
- change grouping (share functions across channels)
- add/remove a KAN layer
- perturb initialization scaling

C. Mamba mutation operators
- change state_dim ± step
- change selectivity_mode (static↔gated)
- change kernel_type (diagonalizable, parameterized)
- add/remove SSM layer

D. Hybrid mutation rules (if individual already hybrid)
- Intra-block mutation applies to each block’s own genes.
- Adapter mutation: change adapter projection size or dropout.
- Family-swap mutation (rare): change family tag at a layer and set new family genes from a default seed.

Mutation probabilities:
- p_mutation_family = 0.2 (per gene)
- p_family_swap = 0.05 overall

Deliverable: mutated offspring.

---

## 7. Offspring evaluation & weight inheritance
- For each offspring:
  - If offspring is minor in-family mutation of a parent: inherit weights from parent (weight inheritance).
  - If offspring is hybrid but retains many layer shapes from one parent: inherit matching layer weights and randomly init adapters/new layers.
  - If offspring is cross-family macro-change: do not inherit incompatible weights (or use distillation/teacher init if feasible).
- Run multi-fidelity evaluation on offspring (Stage 1–3 pipeline).
- Store child fitness metrics.

Deliverable: evaluated offspring with fitness.

---

## 8. Next-generation formation
- Combine elites E + offspring O; perform environmental selection:
  - Option A scalarized: pick top N by scalarized fitness.
  - Option B Pareto: run NSGA-II selection to choose next population of size N (maintains diversity).
- Update per-family elite pools and Pareto archive.
- Update surrogate model with new evaluated data (for later predictions).

Deliverable: new population P_{t+1}.

---

## 9. Repeat & termination
- Repeat evolution loop for G generations (e.g., G = 20–50) or until budget exhausted (GPU-hours).
- Periodically (every K generations) fully train top Pareto models for final evaluation.
- Termination criteria: fixed GPU budget, convergence in Pareto front, or no improvement over M consecutive generations.

Deliverable: final Pareto set of evolved models (pure or hybrid).

---

## 10. Post-processing & compression
- For selected final models, apply EvoPress-style evolutionary compression/pruning to further reduce inference cost while preserving ROUGE.
- Optionally fine-tune pruned models for improved summaries.

Deliverable: compressed deployable models.

---

## 11. Logging, reproducibility, and experiments record
- Save: full genotype, random seeds, training logs, checkpoint (weights), evaluation metrics, FLOPs/time, and full architecture diagrams.
- Version control: git commit hash, environment.yml, and data splits.
- Reproducibility run: re-evaluate best 3 models with fixed seeds and report average metrics.

---

## 12. Risk mitigation and practical tips (for a single GPU)
- Use tiny models (embed_dim 128) for search; scale up only for final candidates.
- Use surrogate predictor + successive halving to avoid wasting compute on bad models.
- Use weight inheritance aggressively for in-family mutations.
- Limit cross-family crossover frequency early in search; increase later if hybrids show promise.
- Start with single-family Transformer-only EA to validate pipeline, then enable KAN/Mamba populations.

---

## Appendix A: What can be exchanged across families (concise list)
- common genes: embed_dim, dropout, layer count (if mapping exists)
- submodule swap: transformer's FFN ↔ KAN-MLP
- layer replacement: replace Transformer block with Mamba SSM block at a specific layer index (with adapter)
- encoder/decoder family swapping (macro-change)
- hyperparameters (numerical values): state_dim, ff_multiplier, n_heads -> used as hints when converting family types

---

## Appendix B: Mutation list (concise)
- Transformer: n_heads, ff_multiplier, n_layers_enc/dec, attention_type, dropout, activation
- KAN: spline_degree, num_basis, groups, n_layers, expansion
- Mamba: state_dim, kernel_type, selectivity_mode, n_layers

---

## Appendix C: Example experimental protocol (recommendation for your first run)
- Population N = 24 (8 per family), µ_elite = 4
- Generations G = 20, offspring per gen = 20
- p_cross_family = 0.1, p_family_swap = 0.05
- Multi-fidelity: 0-proxy → 1-epoch eval → keep top 25% → 3-epoch eval → top 5 full-train
- Log ROUGE-L and tokens/sec; present Pareto front after each run.

---

## Presentation checklist (slides / document)
- Motivation (mixed families + summarization + compute trade-off)
- High-level algorithm flowchart (Init → Eval → Select → Crossover → Mutate → NextGen)
- Family genotypes (tables)
- Cross-family crossover rules (table + examples)
- Fitness definition & multi-fidelity pipeline
- Experimental protocol & compute budget
- Expected deliverables & timeline

---

## Final note (to coordinators)
This workflow balances experimental ambition (mixed-family / hybrid modules) and feasibility (single GPU) by combining family-aware operators, weight inheritance, and multi-fidelity evaluation. It yields interpretable hybrids and a Pareto front of models, ready for downstream compression and deployment.

