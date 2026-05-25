import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from typing import Union
import random
import copy
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import gc
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from code.jamba_model_train import *

# ------------------------------------------------------------
# Demora mais ou menos ~5h30 with 4 epochs on a single gpu and batch size 32
# ------------------------------------------------------------
CHECKPOINT_NAME = "best_model_run_5_seed_7_200_steps.pt"
OUTPUT_CSV = "trained_model_results_run_5_seed_7_200_steps.csv"
MODEL_SAVE_PATH = "trained_model_run_5_seed_7_200_steps.pt"
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 3e-5

def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        # Medição do tempo total
        torch.cuda.synchronize()
        t_inicio = time.time()

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        torch.cuda.synchronize()
        t_fim = time.time()

    # Tempo total em segundos
    tempo_total_s = t_fim - t_inicio
    n_amostras = len(all_labels)   # total de exemplos no dataset de teste

    # Latência média por documento 
    latencia_media_ms = (tempo_total_s / n_amostras) * 1000.0

    # Calcula as métricas de performance
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(loader)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "loss": avg_loss,
        "lat": latencia_media_ms
    }

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Benchmark Final em: {device}")

    # 1. Carregar Dataset
    print("A carregar e tokenizar AG News...")
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Mini-Jamba")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_train = dataset["train"].map(tokenize_function, batched=True).with_format("torch")
    tokenized_test = dataset["test"].map(tokenize_function, batched=True).with_format("torch")
    train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE)

    # 2. Carregar checkpoint e reconstruir o modelo com o genótipo
    print(f"A carregar checkpoint: {CHECKPOINT_NAME}")
    checkpoint = torch.load(CHECKPOINT_NAME, map_location=device)
    genotype = checkpoint['genotype']
    print(f"\n GENÓTIPO DO MODELO: {genotype}")

    config = from_pretrained("TechxGenus/Mini-Jamba")
    base_lm = JambaLM(config, genotype).to(device)
    model = JambaClassifier(base_lm, num_classes=4).to(device)

    # Load with strict=False to handle frozen backbone from evolution
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing_keys:
        print(f"Aviso: {len(missing_keys)} chaves em falta (ex: {missing_keys[:5]}). Serão treinadas de raiz.")
    if unexpected_keys:
        print(f"Aviso: {len(unexpected_keys)} chaves inesperadas (ignoradas).")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo reconstruído com {num_params:,} parâmetros.")

    # 3. Treino 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1*total_steps),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []

    # Start a clock to measure total training time
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        metrics = evaluate(model, test_loader, device, criterion)
        current_f1 = metrics["f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"Novo recorde de F1: {best_f1:.4f}! A guardar pesos...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            print(f"F1 ({current_f1:.4f}) não superou o melhor ({best_f1:.4f}).")

        epoch_data = {
            "epoch": epoch + 1,
            "genotype": "".join(map(str, genotype)),
            "train_loss": avg_train_loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics["acc"],
            "f1_weighted": metrics["f1"],
            "precision": metrics["prec"],
            "recall": metrics["rec"],
            "latency_ms_per_doc": metrics["lat"], #
            "params": num_params
        }
        history.append(epoch_data)
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {metrics['loss']:.4f}")
        print(f"F1: {metrics['f1']:.4f} | Acc: {metrics['acc']:.4f} | Lat: {epoch_data['latency_ms_per_doc']:.2f}ms")

    df_results = pd.DataFrame(history)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBenchmark concluído!")
    print(f"Logs guardados em: {OUTPUT_CSV}")
    print(f"Pesos finais guardados em: {MODEL_SAVE_PATH}")
    print(f"Tempo total de treino: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    train_model()