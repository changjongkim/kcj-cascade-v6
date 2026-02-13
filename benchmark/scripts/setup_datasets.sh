#!/bin/bash
# Script to download external datasets for Cascade V6 Benchmarking

# 1. ShareGPT (Multi-Turn Dialogue) -> for Novelty 1 & 2
# Cleaned JSONL (Small sample)
echo "Downloading ShareGPT (Sample)..."
wget -O benchmark/data_external/sharegpt/sharegpt_cleaned.json \
     https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
# Note: This is an example, actual cleaned ShareGPT requires filtering. 
# Alternatively, use LMSYS-Chat-1M if available.

# 2. PG-19 (Long Context) -> for Novelty 1 & Tiering
# Download a long book (Moby Dick) as a proxy for long context
echo "Downloading PG-19 (Moby Dick)..."
wget -O benchmark/data_external/longbench/moby_dick.txt \
     http://www.gutenberg.org/files/2701/2701-0.txt

# 3. The Stack (Code) -> for Novelty 2 (Dedup) & Locality
# Download a sample source file (e.g. Linux Kernel main.c)
echo "Downloading The Stack (Linux Kernel Main)..."
wget -O benchmark/data_external/thestack/linux_main.c \
     https://raw.githubusercontent.com/torvalds/linux/master/init/main.c

echo "Diverse dataset setup complete."
ls -R benchmark/data_external
