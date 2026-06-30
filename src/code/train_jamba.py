import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import pandas as pd
import time

from jamba_model_train import *

# Config
BATCH_SIZE = 16
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


    tempo_total_s = t_fim - t_inicio
    n_amostras = len(all_labels)

    # Latency
    latencia_media_ms = (tempo_total_s / n_amostras) * 1000.0

    # Performance
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(loader)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "loss": avg_loss,
        "lat": latencia_media_ms
    }


def train_model(genotype_str=None, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing on: {device}")

    # Load Dataset
    print("Loading and tokenizing AG News...")
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Mini-Jamba")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_train = dataset["train"].map(tokenize_function, batched=True).with_format("torch")
    tokenized_test = dataset["test"].map(tokenize_function, batched=True).with_format("torch")
    train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE)

    # Determine genotype
    if checkpoint_path is not None:
        print(f"Load checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        genotype = checkpoint['genotype']
        print(f"\n Genotype: {genotype}")
        state_dict = checkpoint['state_dict']
    elif genotype_str is not None:
        genotype = [int(x) for x in genotype_str.split(',')]
        print(f"\n Genotype (manual): {genotype}")
        state_dict = None   # will train from scratch
    else:
        raise ValueError("Either checkpoint_path or genotype_str must be provided")

    config = from_pretrained("TechxGenus/Mini-Jamba")
    base_lm = JambaLM(config, genotype).to(device)
    model = JambaClassifier(base_lm, num_classes=4).to(device)

    # Load weights only if we have a checkpoint
    if state_dict is not None:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys missing from pretrained weights (ex: {missing_keys[:5]}). Will be trained from scratch.")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys in pretrained weights (ignored).")
    else:
        print("Training from scratch (random initialization).")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model reconstructed with {num_params:,} parameters.")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1*total_steps),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
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

        # saves
        OUTPUT_CSV = "original_jamba_results.csv"
        MODEL_SAVE_PATH = "original_jamba_best.pt"

        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"New F1 record: {best_f1:.4f}! Saving weights...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            print(f"F1 ({current_f1:.4f}) did not surpass the best ({best_f1:.4f}).")

        epoch_data = {
            "epoch": epoch + 1,
            "genotype": "".join(map(str, genotype)),
            "train_loss": avg_train_loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics["acc"],
            "f1_weighted": metrics["f1"],
            "precision": metrics["prec"],
            "recall": metrics["rec"],
            "latency_ms_per_doc": metrics["lat"],
            "params": num_params
        }
        history.append(epoch_data)
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {metrics['loss']:.4f}")
        print(f"F1: {metrics['f1']:.4f} | Acc: {metrics['acc']:.4f} | Lat: {epoch_data['latency_ms_per_doc']:.2f}ms")

    df_results = pd.DataFrame(history)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBenchmark concluded!")
    print(f"Logs saved to: {OUTPUT_CSV}")
    print(f"Final weights saved to: {MODEL_SAVE_PATH}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype", type=str, default=None, help="Genotype as comma-separated list (e.g., '0,0,0,0,1,0,...')")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint")
    args = parser.parse_args()
    train_model(genotype_str=args.genotype, checkpoint_path=args.checkpoint)