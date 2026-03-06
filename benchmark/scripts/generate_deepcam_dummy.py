import h5py
import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import time

def create_stats_file(path):
    with h5py.File(path, 'w') as f:
        grp = f.create_group("climate")
        # MLPerf DeepCAM requires 16 channels, maxval > minval to avoid division by zero
        grp.create_dataset("minval", data=np.zeros(16, dtype=np.float32))
        grp.create_dataset("maxval", data=np.ones(16, dtype=np.float32))

def create_sample(filepath):
    data_shape = (768, 1152, 16)
    label_shape = (768, 1152)

    # Use fixed seed per file or deterministic data to keep it fast
    # Random data is used to avoid extreme fast compression if any storage layer attempts it
    np.random.seed(hash(os.path.basename(filepath)) % (2**32))
    
    data = np.random.rand(*data_shape).astype(np.float32)
    labels = np.random.randint(0, 3, size=label_shape, dtype=np.int64)
    
    with h5py.File(filepath, 'w') as f:
        grp = f.create_group("climate")
        grp.create_dataset("data", data=data)
        grp.create_dataset("labels_0", data=labels)
    
    return filepath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_train", type=int, default=256)
    parser.add_argument("--num_val", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    stats_file = os.path.join(args.out_dir, "stats.h5")
    print(f"Creating stats file at {stats_file}")
    create_stats_file(stats_file)
    
    print(f"Creating {args.num_train} train samples and {args.num_val} val samples...")
    print(f"Each file is roughly ~60MB.")
    
    train_paths = [os.path.join(train_dir, f"data_{i:05d}.h5") for i in range(args.num_train)]
    val_paths = [os.path.join(val_dir, f"data_{i:05d}.h5") for i in range(args.num_val)]
    all_paths = train_paths + val_paths
    
    start_time = time.time()
    
    # Generate concurrently
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for i, _ in enumerate(executor.map(create_sample, all_paths)):
            if (i + 1) % 50 == 0:
                print(f" -> Generated {i + 1}/{len(all_paths)} files...")
                
    elapsed = time.time() - start_time
    total_size_gb = (len(all_paths) * 56.6) / 1024
    
    print(f"✅ Dummy dataset creation complete in {elapsed:.1f} seconds! (Total: {total_size_gb:.2f} GB)")

if __name__ == "__main__":
    main()
