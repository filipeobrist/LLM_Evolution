# Plano B: Trivial
## 1. Initial Population Design
- A população inicial será composta por pure transformers, KANs e mambas, em que N_T=N_K=N_M
- Para começar, talvez uma população pequena para testar mais rápido tipo 50.
- Genes em comum de todas as familias: d_model, depth, dropout, norm_type, context_len
- Cada familia tem também hiperparametros próprios

## 2. Genes
- **Transformer:**
  - Depth (int): *num_layers*, 6-64
  - Hidden size (int): *d_model*, 256-4096
  - Attention heads (int and must divide d_model): *num_heads*, 4-64
  - FFN inner dim (int): *d_ff*, 4*-8*d_model
  - MLP activation (cat): *activation*,  {gelu, swiglu, geglu}
  - Dropout in attention (float): *attn_dropout*, 0-0.2
  - Dropout after residual (float): *resid_dropout*, 0-0.2
  - Positional encoding type (cat): *pos_enc*, {rope, alibi, learned}
  - Normalization variant (cat): *norm_type*, {layernorm, rmsnorm}
  - Speed/efficiency form (cat): *attn_variant*, {vanilla, flash, grouped, multiquery}
  - Sequence length/ Input context (int): *context_len*, 512-32768
  - Share embedding & output weights (bool): *weight_tie*, True/False

- **Mamba:**
  - *num_layers*, *d_model*, *norm_type*, *context_len*, *dropout*
  - SSM state size (int): *d_state*, 16-512
  - Order of SSM kernel (int): *ssm_order*, 1-4
  - Rank for Δt parameterization (int): *dt_rank*, 1-64
  - Local conv pre-filter size (int): *conv_kernel*, 1-15
  - Nonlinearity (cat): *activation*, {silu, gelu}
  - Include bias in state updates (bool): *bias*, True/False
  - Residual scaling (helps deep SSMs) (float): *resid_scale*, 0.5-1

- **KAN:**
  - *num_layers*, *d_model*, *norm_type* (just layernorm or none), *context_len*, *dropout*
  - Number of group-KANs (int): *k_groups*, 2-16
  - Basis type (cat): *basis_funcs*, {chebyshev, legendre, bspline, fourier}
  - Polynomial/trig order per dimension (int): *basis_order*, 2-8
  - Post-basis nonlinearity (cat): *activation*, {none, relu, silu}
  - Residual skip (bool): *skip_connection*, True/False

## 3. Mutation
- Common mutations (for all). This is the mutation option for the hybrid family
  - Add / remove layer (±1–3)
  - Scale model width (d_model *= random.choice([0.75, 1.0, 1.25]))
  - Perturb learning rate, dropout, activation
  - Change normalization type
  - Change context length (double or halve)

- Family specific mutations
  - **Transformer**: Change attn_variant; swap positional encoding; change num_heads keeping divisibility; toggle weight_tie
  - **Mamba**: Mutate d_state, ssm_order, dt_rank, or conv_kernel. Increase/decrease residual scale slightly.
  - **KAN**: Mutate k_groups or basis_order; switch basis_funcs; toggle skip connection.

- Mutation magnitudes should be modest to preserve trainability

## 4. Crossover Logic
- Crossover operates on the gene level and must handle both intra-family and inter-family matings.

### A. Intra-family crossover (same architecture type)
- parameters are compatible
- Pseudo-code:
  ```
  def crossover_same_family(parentA, parentB):
      child = {}
      for gene in parentA.keys():
          if random.random() < 0.5:
              child[gene] = parentA[gene]
          else:
              child[gene] = parentB[gene]
      return child
  ```
- for numeric genes, we use interpolation: child[g] = α*A[g] + (1−α)*B[g]

### B. Cross-family crossover
- Shared global genes (d_model, depth, dropout, norm_type, context_len): blend or copy as in intra-family crossover.
- Unique local genes: keep from each parent’s block but allow conceptual analog swaps
- Pseudocode:
  ```
  def crossover_cross_family(parentA, parentB):
    child = {}
    # global shared genes
    for gene in shared_global_genes:
        if is_numeric(gene):
            α = random.uniform(0.3, 0.7)
            child[gene] = α*parentA[gene] + (1-α)*parentB[gene]
        else:
            child[gene] = random.choice([parentA[gene], parentB[gene]])

    # choose dominant family
    child['family'] = random.choice([parentA['family'], parentB['family']])

    # take family-specific genes from dominant parent, 
    # but map analogous concepts when possible
    if child['family'] == "MAMBA":
        child['d_state'] = int((parentA.get('d_ff', parentB.get('d_state',256)) / child['d_model']) * 64)
        child['ssm_order'] = random.choice([1,2,3])
    elif child['family'] == "KAN":
        child['k_groups'] = int(np.clip(child['d_model']/128,2,16))
        child['basis_order'] = random.choice([3,4,5,6])
    elif child['family'] == "TRANSFORMER":
        child['num_heads'] = round(child['d_model']/64)
        child['d_ff'] = 4*child['d_model']
    return child
  ```

- Basically, there is no hybrid. The child "chooses" the family he wants from the parents, mapping when possible analougs concept


## 5. Fitness 
- Balance between performance and cost