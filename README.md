# LLM_Evolution

An evolutionary algorithm that explores the design space of hybrid Transformer–Mamba architectures for text classification.

## The Overview

This project investigates whether a genetic algorithm can automatically discover efficient combinations of Mamba (State Space Model) and Transformer (attention) layers. A variable‑length binary genotype specifies the type of each layer, and the fitness of an architecture is measured by training it from scratch on a downstream task.

The framework supports two text classification tasks:
- **AG News** (English, 4 classes)
- **PROPOR FOS Classification** (Portuguese, 5 classes)


## Requirements

- Python 3.8+
- PyTorch 1.13+
- Hugging Face `transformers` and `datasets`
- scikit-learn
- mamba‑ssm (`pip install mamba-ssm`)
- matplotlib, numpy, pandas

## Quick Start

1. **Run the evolutionary search AG News**
   ```bash
   python run_evolution.py
    ```

2. **Intensive training of the best architecture**
    ```bash 
    python train_model.py
    ```

Parameters are defined at the top of each script and can be easily adjusted