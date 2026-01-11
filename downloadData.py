import os
from datasets import load_dataset

dataset_dir = "./datasets"

wikitext_dir = os.path.join(dataset_dir, "wikitext")
print("Downloading WikiText-2...")
try:
    dataset = load_dataset(
        "wikitext", 
        "wikitext-2-raw-v1", 
        cache_dir=wikitext_dir 
    )
    print(f"WikiText-2 saved to: {wikitext_dir}")
    print(f"Train samples: {len(dataset['train'])}")
except Exception as e:
    print(f"Failed to download WikiText-2: {e}")

pg19_dir = os.path.join(dataset_dir, "pg19_sample")
print("\nDownloading PG-19 sample...")
try:
    dataset = load_dataset(
        "emozilla/pg19", 
        split="train", 
        streaming=True, 
        cache_dir=pg19_dir
    )
    sample = next(iter(dataset))
    sample_file = os.path.join(pg19_dir, "pg19_sample.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample["text"])
    print(f"PG-19 sample saved to: {sample_file}")
    print(f"Text length: {len(sample['text'])} characters")
except Exception as e:
    print(f"Failed to download PG-19: {e}")