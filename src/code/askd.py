import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

# load the dataset
ds = load_dataset("ivosimoes/PROPOR_FOS_Classification", split="train")
tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Mini-Jamba")   # correct tokenizer

df = pd.DataFrame(ds)
df['text'] = df.apply(lambda row: f"Título: {row['title']}\nPalavras-chave: {row['keywords']}\nResumo: {row['abstract']}", axis=1)
texts = df['text'].tolist()

lengths = [len(tokenizer.encode(text, truncation=False, max_length=9999)) for text in texts]
lengths = np.array(lengths)

# Percentiles
print(f"95th percentile: {np.percentile(lengths, 95):.0f}")
print(f"90th percentile: {np.percentile(lengths, 90):.0f}")
print(f"75th percentile: {np.percentile(lengths, 75):.0f}")
print(f"Median: {np.median(lengths):.0f}")

# Fraction longer than 512
fraction_longer = np.mean(lengths > 512) * 100
print(f"Percentage of samples longer than 512 tokens: {fraction_longer:.1f}%")