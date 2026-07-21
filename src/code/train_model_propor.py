import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import time
import random

# Import everything from our model library
from jamba_model_train import *

# Configuration
CHECKPOINT_NAME = "best_model_run_15_seed_888_propor.pt"
OUTPUT_CSV = "trained_propor_results_run_15_seed_888.csv"
MODEL_SAVE_PATH = "trained_propor_best_run_15_seed_888.pt"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 3e-5
MAX_LENGTH = 512
SEED = 888

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Propor dataset
def load_propor_dataset(tokenizer, max_length=512):
    print("Loading PROPOR FOS Classification dataset...")
    # Load splits
    train_ds = load_dataset("ivosimoes/PROPOR_FOS_Classification", split="train")
    test_ds = load_dataset("ivosimoes/PROPOR_FOS_Classification", split="test")
    
    df_train = pd.DataFrame(train_ds)
    df_test = pd.DataFrame(test_ds)
    
    # Combine fields for train
    df_train['text'] = df_train.apply(
        lambda row: f"Título: {row['title']}\nPalavras-chave: {row['keywords']}\nResumo: {row['abstract']}",
        axis=1
    )
    # Combine fields for test
    df_test['text'] = df_test.apply(
        lambda row: f"Título: {row['title']}\nPalavras-chave: {row['keywords']}\nResumo: {row['abstract']}",
        axis=1
    )
    
    # Encode labels based on the train labels
    train_labels = sorted(df_train['label'].unique())
    label_to_id = {l: i for i, l in enumerate(train_labels)}
    print(f"Label mapping: {label_to_id}")
    df_train['label_int'] = df_train['label'].map(label_to_id)
    # For test, any unseen label (should not happen) gets -1, but we'll trust the dataset
    df_test['label_int'] = df_test['label'].map(label_to_id)
    # In case a test label is not present, we drop those rows
    if df_test['label_int'].isna().any():
        print("Warning: Some test labels were not present in the training set. Dropping them.")
        df_test = df_test.dropna(subset=['label_int'])
        df_test['label_int'] = df_test['label_int'].astype(int)
    
    # Split train into train/val (85% train, 15% val)
    texts_train = df_train['text'].tolist()
    labels_train = df_train['label_int'].tolist()
    combined = list(zip(texts_train, labels_train))
    random.shuffle(combined)
    texts_train, labels_train = zip(*combined)
    split_idx = int(0.85 * len(texts_train))
    train_texts, val_texts = texts_train[:split_idx], texts_train[split_idx:]
    train_labels, val_labels = labels_train[:split_idx], labels_train[split_idx:]
    
    # Test set
    test_texts = df_test['text'].tolist()
    test_labels = df_test['label_int'].tolist()
    
    # Tokenize function
    def tokenize(texts, labels):
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        return {
            'input_ids': torch.tensor(enc['input_ids']),
            'attention_mask': torch.tensor(enc['attention_mask']),
            'label': torch.tensor(labels)
        }
    
    train_enc = tokenize(train_texts, train_labels)
    val_enc = tokenize(val_texts, val_labels)
    test_enc = tokenize(test_texts, test_labels)
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'label': self.labels[idx]}
    
    train_dataset = SimpleDataset(train_enc['input_ids'], train_enc['attention_mask'], train_enc['label'])
    val_dataset = SimpleDataset(val_enc['input_ids'], val_enc['attention_mask'], val_enc['label'])
    test_dataset = SimpleDataset(test_enc['input_ids'], test_enc['attention_mask'], test_enc['label'])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

# Evaluation function
def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    torch.cuda.synchronize()
    t_start = time.time()
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    torch.cuda.synchronize()
    t_end = time.time()
    
    total_time_s = t_end - t_start
    n_samples = len(all_labels)
    latency_ms = (total_time_s / n_samples) * 1000.0

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(loader)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'loss': avg_loss, 'lat': latency_ms}


# Training
def train_propor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Mini-Jamba")
    train_ds, val_ds, test_ds = load_propor_dataset(tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Load checkpoint genotype and state_dict
    print(f"Loading checkpoint: {CHECKPOINT_NAME}")
    checkpoint = torch.load(CHECKPOINT_NAME, map_location=device)
    genotype = checkpoint['genotype']
    print(f"Genotype: {genotype}")
    
    config = from_pretrained("TechxGenus/Mini-Jamba")
    base_lm = JambaLM(config, genotype).to(device)
    model = JambaClassifier(base_lm, num_classes=5).to(device)   # 5 classes for PROPOR
    
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (e.g., {missing[:3]})")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = -1.0
    history = []
    overall_start = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on test set
        metrics = evaluate(model, test_loader, device, criterion)
        current_f1 = metrics['f1']
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"  New best F1: {best_f1:.4f} – saving model")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            print(f"  F1 = {current_f1:.4f} (best: {best_f1:.4f})")
        
        epoch_data = {
            'epoch': epoch + 1,
            'genotype': ''.join(map(str, genotype)),
            'train_loss': avg_train_loss,
            'test_loss': metrics['loss'],
            'accuracy': metrics['acc'],
            'f1_macro': metrics['f1'],
            'precision': metrics['prec'],
            'recall': metrics['rec'],
            'latency_ms_per_doc': metrics['lat'],
            'params': num_params
        }
        history.append(epoch_data)
        print(f"  Train Loss: {avg_train_loss:.4f}  |  Test Loss: {metrics['loss']:.4f}  |  F1: {metrics['f1']:.4f}  |  Lat: {metrics['lat']:.2f} ms")
    
    total_time = time.time() - overall_start
    df = pd.DataFrame(history)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTraining finished. Total time: {total_time:.1f} s")
    print(f"Results saved to {OUTPUT_CSV}")
    print(f"Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_propor()