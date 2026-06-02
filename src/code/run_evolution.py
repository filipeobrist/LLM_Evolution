import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from typing import Union
import random
import copy
from transformers import AutoTokenizer
import gc
from torch.utils.data import DataLoader
# from tqdm import tqdm
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Count the execution time
import time

from jamba_model_evolve import *

# Configurations start here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAMBA, ATTN = 0, 1
MIN_LAYERS, MAX_LAYERS = 4, 20
NUM_CLASSES = 4
MUTATION_RATE = 0.5
MUTATION_RATE_STRUCTURAL = 0.25
CROSSOVER_RATE = 0.8 
POP_SIZE = 30
GENERATIONS = 200
ELITISM = 1
LEARNING_RATE = 4e-4
FINE_TUNE = True

DATALOADER_BASE_SEED = 42

BATCH_SIZE = 16
STEPS_1 = 600


def load_agnews(tokenizer, n_train=60000, n_val=1500):
    """Loads the AG News dataset and pre-tokenizes it using the provided tokenizer. It returns tokenized train and validation datasets ready for PyTorch."""
    print("Pre-tokenizing dataset...")
    ds = load_dataset("ag_news")
    
    def tokenize_function(examples):
        # Pre-Tokenization
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    # Split
    train = ds["train"].shuffle(seed=42).select(range(n_train))
    val = ds["test"].shuffle(seed=42).select(range(n_val))
    
    tokenized_train = train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val.map(tokenize_function, batched=True, remove_columns=["text"])
    
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_train, tokenized_val


def train_model(model, train_ds, steps, val_ds=None, patience=3, gen0=False, ind_id="0", dl_seed=None):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    if dl_seed is not None:
        g = torch.Generator()
        g.manual_seed(dl_seed)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    else:
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    best_f1 = 0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    val_check_interval = 100 
    
    # Listas para guardar métricas apenas se gen0 for True para não gastar RAM à toa
    train_losses = [] if gen0 else None
    val_f1s = [] if gen0 else None
    
    start_time = time.time()
    step_count = 0
    
    while step_count < steps:
        for batch in loader:
            if step_count >= steps: break
            
            input_ids, labels = batch["input_ids"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Só guarda a loss se estivermos na Geração 0
            if gen0:
                train_losses.append(loss.item())
            
            step_count += 1
            
            # Verificação de Validação & Early Stopping
            if step_count % val_check_interval == 0 and val_ds is not None:
                current_f1, _ = evaluate_model(model, val_ds)
                model.train()
                
                if gen0:
                    val_f1s.append((step_count, current_f1))
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                #
                if no_improve_count >= patience:
                    model.load_state_dict(best_weights)
                    if gen0:
                        save_training_plot(train_losses, val_f1s, ind_id)
                    return time.time() - start_time, train_losses
    
    # Se chegarmos ao fim dos steps sem disparar o Early Stopping
    if gen0:
        save_training_plot(train_losses, val_f1s, ind_id)
        
    return time.time() - start_time, train_losses

def save_training_plot(losses, f1s, ind_id):
    """Gera um gráfico da evolução do treino."""
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(losses, color='tab:red', alpha=0.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val F1', color='tab:blue')
    steps, f1_values = zip(*f1s) if f1s else ([], [])
    ax2.plot(steps, f1_values, color='tab:blue', marker='o', label='F1')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Training Progress - Indivíduo {ind_id}')
    plt.savefig(f'../plots/training_plot_{ind_id}.png')
    plt.close()


# Utils
@torch.no_grad()
def evaluate_model(model, val_ds):
    """Returns F1 score and average latency per sample on the validation set."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    all_preds, all_labels, latencies = [], [], []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"] # Mantém no CPU para facilitar
        
        torch.cuda.synchronize() 
        start_lat = time.perf_counter()
        logits = model(input_ids)
        torch.cuda.synchronize() 
        latencies.append(time.perf_counter() - start_lat)
        
        preds = torch.argmax(logits, dim=-1).cpu() 
        
   
        all_preds.extend(preds.numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
            
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Latência média por amostra
    avg_lat = (sum(latencies) / len(latencies)) / BATCH_SIZE
    
    return f1, avg_lat


def get_trainable_state_dict(model):
    """Returns a state_dict containing only parameters that require gradients."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items() if v.requires_grad}


def setup_model_trainability(model, full_fine_tune=False):
    """
    If full_fine_tune is True: Unfreezes everything for max accuracy.
    If full_fine_tune is False: Only unfreezes Norms, MoE, and Classifier.
    """
    if full_fine_tune:
        # print("High-Performance Mode: Unfreezing all parameters...")
        for p in model.parameters():
            p.requires_grad = True
    else:
        # strategy for low VRAM
        for name, p in model.named_parameters():
            if (
                "classifier" in name
                or "layernorm" in name.lower()
                or "moe" in name.lower()
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False


# --- GENETIC OPERATORS ---

def generate_random_genotype(min_layers=MIN_LAYERS):
    length = random.randint(min_layers, MAX_LAYERS)
    return [random.randint(0, 1) for _ in range(length)]

def crossover(g1, g2):
    """Crossover de ponto único para genótipos de tamanhos variáveis"""
    point = random.randint(1, min(len(g1), len(g2)) - 1)
    # Combina a primeira parte de um com a segunda do outro
    child_g = g1[:point] + g2[point:]
    
    # Garante que o filho respeita os limites de profundidade
    if len(child_g) > MAX_LAYERS:
        child_g = child_g[:MAX_LAYERS]
    elif len(child_g) < MIN_LAYERS:
        # Se for muito pequeno, adiciona camadas aleatórias até ao mínimo
        while len(child_g) < MIN_LAYERS:
            child_g.append(random.randint(0, 1))
            
    return child_g

def mutate_bitflip(genotype):
    """Flip each gene with probability 1 / len(genotype)."""
    new_gen = list(genotype)
    rate = 1.0 / len(new_gen)
    for i in range(len(new_gen)):
        if random.random() < rate:
            new_gen[i] = 1 - new_gen[i]
    return new_gen

def mutate_structural(genotype, insert_prob=0.5):
    """
    Either insert a random gene or delete a random gene. The choice between insert/delete
    is balanced by insert_prob.
    """
    new_gen = list(genotype)
    if len(new_gen) < MAX_LAYERS and (len(new_gen) <= MIN_LAYERS or random.random() < insert_prob):
        # insert a random gene
        new_gen.insert(random.randint(0, len(new_gen)), random.randint(0, 1))
    elif len(new_gen) > MIN_LAYERS:
        # delete a random gene
        del new_gen[random.randint(0, len(new_gen)-1)]
    # if both conditions fail (i.e. at min and trying to delete, or at max and trying to insert), do nothing
    return new_gen

# SELECTION

def tournament_selection(population, k=3):
    # 1. Pick k random individuals from the whole population
    selection_pool = random.sample(population, k)
    
    # 2. The winner is the one with the best fitness
    winner = max(selection_pool, key=lambda x: x['fitness'])
    
    return winner

# GENOTYPE 

def apply_genotype(model, genotype):
    """Applies the genotype to the model by activating/deactivating layers and setting them to Mamba or Attention based on the gene value."""
    for i, layer in enumerate(model.lm.jamba.layers):
        if i < len(genotype):
            layer.active = True
            layer.use_mamba = (genotype[i] == MAMBA)
            layer.use_attention = (genotype[i] == ATTN)
        else:
            layer.active = False

def evaluate_individual(config, genotype, train_ds, val_ds, steps, 
                        inherited_weights=None, parent_genotype=None, 
                        gen0=False, ind_id="0", dl_seed=None):
    """
    Build a JambaLM with the given genotype, optionally inherit weights for
    layers that have the same type as the parent, then fine_tune and evaluate.
    """
    # Create model from genotype
    model = JambaLM(config, genotype).to(DEVICE)
    
    if inherited_weights is not None and parent_genotype is not None:
        # Turned off for now
        weight_share(model, inherited_weights, genotype, parent_genotype)
    
    classifier = JambaClassifier(model, NUM_CLASSES).to(DEVICE)
    setup_model_trainability(classifier, full_fine_tune=FINE_TUNE)
    
    train_time, losses = train_model(classifier, train_ds, steps, 
                                     val_ds=val_ds, patience=3, 
                                     gen0=gen0, ind_id=ind_id, dl_seed=dl_seed)
    f1, latency = evaluate_model(classifier, val_ds)
    
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    stats = {
        'f1': f1, 'latency': latency, 
        'params': total_params,
        'depth': len(genotype), 'train_time': train_time
    }
    
    print(f"Evaluated Genotype: {genotype} | F1: {f1:.4f} | Latency: {latency:.4f}s | Train Time: {train_time:.2f}s")

    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return get_trainable_state_dict(classifier), stats, losses

def fitness(population_list):
    if not population_list: return

    for ind in population_list:
        stats = ind['stats']
        f1 = stats['f1']
        
        base_score = f1

        # lat_ratio =  BASELINE_LATENCY / stats['latency']
        # param_ratio = BASELINE_PARAMS / stats['params']  
        
        # penalty = 1.0 * lat_ratio + 0.5 * param_ratio # Ver o porque de eles usarem o exp 

        ind['fitness'] = base_score # * penalty



def weight_share(child_model, parent_weights, child_genotype, parent_genotype):
    # Errado, porque o modelo filho pode ter camadas a mais ou a menos, e o índice da camada no filho pode não
    #  corresponder ao do pai. Precisamos de uma lógica que mapeie camadas do filho para camadas do pai 
    # com base no tipo (Mamba vs Attention) e na posição relativa, não apenas no índice absoluto.
    child_dict = child_model.state_dict()
    new_weights = {}

    for name, param in child_dict.items():
        # Se o peso não for de uma camada evoluível
        if "layers" not in name:
            if name in parent_weights:
                new_weights[name] = parent_weights[name]
            continue
        
        # Extrair o índice da camada
        parts = name.split('.')
        layer_idx = int(parts[parts.index("layers") + 1])
        
        # Só herdar se a camada existir no pai E for do mesmo tipo
        if layer_idx < len(parent_genotype):
            if child_genotype[layer_idx] == parent_genotype[layer_idx]:
                if name in parent_weights:
                    new_weights[name] = parent_weights[name]
    

    child_model.load_state_dict(new_weights, strict=False)

def plot_population_vs_best(data):
    plt.figure(figsize=(12, 6))
    
    # Extrair todas as curvas de loss
    all_curves = [d['losses'] for d in data]

    max_len = STEPS_1
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in all_curves])
    
    # Calcular Média
    mean_loss = np.mean(padded, axis=0)
    
    # Identificar o Melhor Indivíduo (por F1)
    best_idx = np.argmax([d['f1'] for d in data])
    best_curve = padded[best_idx]
    
    # Plot
    steps = np.arange(max_len)
    plt.plot(steps, mean_loss, label='Population Average', color='black', linewidth=2, linestyle='--')
    plt.plot(steps, best_curve, label=f'Best model (F1: {data[best_idx]["f1"]:.4f})', color='blue', linewidth=2)
    

    plt.fill_between(steps, np.min(padded, axis=0), np.max(padded, axis=0), color='gray', alpha=0.1, label='Range da População')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Convergência na Geração 0: Média vs Melhor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../plots/gen0_population_analysis.png')
    plt.close()

# EVOLUTION LOOP 
def evolve(base_model, train_ds, val_ds, pop_size=30, generations=100, elitism=1, run_name="evolution_run"):
    history_logs = []
    gen0_data = []
    best_overall_f1 = -1.0
    best_overall_data = {}
    
    # Geração 0 
    gen0_seed = DATALOADER_BASE_SEED + 0
    gen_start_time = time.time()
    print(f"--- Generation 0: Initializing {pop_size} individuals ---")
    population = []
    for i in range(pop_size):
        g = generate_random_genotype(min_layers=10)
        # Treino do zero para a base inicial
        w, s, losses = evaluate_individual(base_model.config, g, train_ds, val_ds, STEPS_1, inherited_weights=None, gen0=True, ind_id=str(i), dl_seed=gen0_seed)
        population.append({'genotype': g, 'weights': w, 'stats': s, 'losses': losses})

    fitness(population)
    population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    gen0_data = [{'losses': ind['losses'], 'f1': ind['stats']['f1']} for ind in population]

    current_gen_best = population[0]

    gen_duration = (time.time() - gen_start_time) / 60

    for i, ind in enumerate(population):
        history_logs.append({
            'generation': 0, 'rank': i, 'fitness': ind['fitness'],
            'f1': ind['stats']['f1'], 'latency': ind['stats']['latency'],
            'params': ind['stats']['params'], 'depth': ind['stats']['depth'],
            'gen_time_min': gen_duration, 'genotype': str(ind['genotype'])
        })

    pd.DataFrame(history_logs).to_csv(f"results_{run_name}.csv", index=False)
    print(f"Gen 0 Best: F1 {population[0]['stats']['f1']:.4f}")

    plot_population_vs_best(gen0_data)
    

    # LOOP DE EVOLUÇÃO
    for gen in range(1, generations):
        gen_start_time = time.time()
        new_candidates = population[:elitism]

        print(f"--- Generation {gen}: Evolving population ---")
        
        gen_seed = DATALOADER_BASE_SEED + gen
        while len(new_candidates) < pop_size:
            # SELEÇÃO E CROSSOVER
            if random.random() < CROSSOVER_RATE:
                p1, p2 = tournament_selection(population), tournament_selection(population)
                child_g = crossover(p1['genotype'], p2['genotype'])
            else:
                # Se não há crossover, clonamos um progenitor
                parent = tournament_selection(population)
                child_g = list(parent['genotype'])
            
            # Mutation
            if random.random() < MUTATION_RATE:
                child_g = mutate_bitflip(child_g)
            if random.random() < MUTATION_RATE_STRUCTURAL:
                child_g = mutate_structural(child_g)

            # Avaliação do filho
            w, s, _ = evaluate_individual(base_model.config, child_g, train_ds, val_ds, STEPS_1, inherited_weights=None, dl_seed=gen_seed)
            
            new_candidates.append({'genotype': child_g, 'weights': w, 'stats': s})

        population = new_candidates
        fitness(population)
        population = sorted(population, key=lambda x: x['fitness'], reverse=True)

        current_gen_best = population[0]

        if current_gen_best['stats']['f1'] > best_overall_f1:
            best_overall_f1 = current_gen_best['stats']['f1']
            # Guardamos
            best_overall_data = {
                'genotype': current_gen_best['genotype'],
                'state_dict': copy.deepcopy(current_gen_best['weights']),
                'fitness': current_gen_best['fitness'],
                'f1': current_gen_best['stats']['f1'],
                'latency': current_gen_best['stats']['latency'],
                'params': current_gen_best['stats']['params'],
                'depth': current_gen_best['stats']['depth'],
                'generation': gen
            }
            print(f"Novo Recorde no Run! Gen {gen}: F1 {best_overall_f1:.4f}")

        # Logs e Monitorização
        gen_duration = (time.time() - gen_start_time) / 60
        for i, ind in enumerate(population):
            history_logs.append({
                'generation': gen, 'rank': i, 'fitness': ind['fitness'],
                'f1': ind['stats']['f1'], 'latency': ind['stats']['latency'],
                'params': ind['stats']['params'], 'depth': ind['stats']['depth'],
                'gen_time_min': gen_duration, 'genotype': str(ind['genotype'])
            })

        pd.DataFrame(history_logs).to_csv(f"results_{run_name}.csv", index=False)
        print(f"Gen {gen} Best: F1 {population[0]['stats']['f1']:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()

    return best_overall_data



SEEDS = [42, 123, 999, 2024, 7]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Garante que as operações no GPU sejam determinísticas
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Clean up before starting
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")



tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Mini-Jamba")
train_ds, val_ds = load_agnews(tokenizer)

# We load the base model once and move it to the device
# It stays "frozen" as a template; deepcopy will be used for individuals
base_model = from_pretrained("TechxGenus/Mini-Jamba").to(device)

# Run
def run_thesis_experiment():
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*30}")
        print(f"Iniciating RUN {run_idx + 1}/5 (SEED: {seed})")
        print(f"{'='*30}")
            
        # Preparar ambiente para este run
        set_seed(seed)
        run_name = f"run_{run_idx+1}_seed_{seed}_600_steps"
            
        # Reinicializar tudo para evitar "leak" de memória entre runs
        gc.collect()
        torch.cuda.empty_cache()

        best_of_run = evolve(
            run_name=run_name,
            base_model=base_model,
            train_ds=train_ds,
            val_ds=val_ds,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            elitism=ELITISM
        )
        
        checkpoint_path = f"best_model_{run_name}.pt"
        torch.save({
            'genotype': best_of_run['genotype'],
            'state_dict': best_of_run['state_dict'],
            'f1': best_of_run['f1'],
            'seed': seed,
            'generation': best_of_run['generation']
        }, checkpoint_path)
        
        print(f"Run {run_idx+1} concluído. Melhor F1: {best_of_run['f1']:.4f} (Gen {best_of_run['generation']})")
        print(f"Modelo guardado em: {checkpoint_path}")

    
if __name__ == "__main__":
    run_thesis_experiment()