# Search space for the summarization-oriented Transformer (encoder–decoder)
TRANSFORMER_SPACE = {
    # Shared architecture parameters
    "embed_dim": [128, 256, 512],
    "n_heads": [2, 4, 8],
    "ff_multiplier": [2, 4, 6],
    "dropout": [0.0, 0.1, 0.2],

    # Encoder/decoder depth (separate for flexibility)
    "n_layers_enc": [2, 4, 6, 8],
    "n_layers_dec": [2, 3, 4, 6],

    # Optional: you can let NAS explore sequence length and max token context
    "max_len": [256, 512, 1024],

    # Activation type (still relevant for feed-forward sublayer)
    "activation": ["gelu", "relu"],
}
