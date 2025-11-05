"""
Search space for KAN models (discrete choices).
Tune these ranges based on your compute capacity.
"""
KAN_SPACE = {
    "embed_dim": [128, 256, 512],
    "enc_layers": [2, 4, 6],
    "dec_layers": [1, 2, 3],
    "mlp_hidden": [256, 512, 1024],
    "n_heads": [2, 4, 8],
    "dropout": [0.0, 0.1, 0.2],
    # If real KAN supports more fine-grained spline params, add them when KANLinear is available:
    # "spline_degree": [3, 5, 7],
    # "num_basis": [8, 16, 32],
}
