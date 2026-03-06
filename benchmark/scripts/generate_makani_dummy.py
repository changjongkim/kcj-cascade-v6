import h5py
import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import time
import datetime as dt
import json

def create_makani_sample(args):
    filepath, year, num_samples, num_channels, img_size_h, img_size_w = args
    # We will generate it chunk by chunk directly into the file to save RAM
    
    rng = np.random.default_rng(seed=hash(filepath) % (2**32))
    
    channel_names = [f"chan_{idx}" for idx in range(num_channels)]
    longitude = np.linspace(0, 360, img_size_w, endpoint=False)
    latitude = np.linspace(-90, 90, img_size_h, endpoint=True)[::-1]
    
    dhours = (365 * 24) // num_samples
    
    with h5py.File(filepath, "w") as hf:
        dset = hf.create_dataset("fields", shape=(num_samples, num_channels, img_size_h, img_size_w), dtype=np.float32)
        
        # Write chunk by chunk (e.g., 10 samples at a time)
        chunk_size = max(1, 1000 // (num_channels)) # Avoid huge RAM spikes
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            batch = rng.random((end_idx - i, num_channels, img_size_h, img_size_w), dtype=np.float32)
            dset[i:end_idx] = batch
            
        # Annotations
        year_start = dt.datetime(year=year, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
        timestamps = year_start + np.arange(0, 365 * 24 * 3600, dhours * 3600, dtype=np.float64)[:num_samples]
        
        hf.create_dataset("timestamp", data=timestamps)
        hf.create_dataset("channel", len(channel_names), dtype=h5py.string_dtype(length=max(len(c) for c in channel_names)))
        hf["channel"][...] = channel_names
        hf.create_dataset("lat", data=latitude)
        hf.create_dataset("lon", data=longitude)
        
        # Attach scales
        hf["timestamp"].make_scale("timestamp")
        hf["channel"].make_scale("channel")
        hf["lat"].make_scale("lat")
        hf["lon"].make_scale("lon")
        
        hf["fields"].dims[0].attach_scale(hf["timestamp"])
        hf["fields"].dims[1].attach_scale(hf["channel"])
        hf["fields"].dims[2].attach_scale(hf["lat"])
        hf["fields"].dims[3].attach_scale(hf["lon"])

    return filepath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples_per_year", type=int, default=1460)  # 4x per day
    parser.add_argument("--num_channels", type=int, default=20)
    parser.add_argument("--img_size_h", type=int, default=720) 
    parser.add_argument("--img_size_w", type=int, default=1440)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    test_path = os.path.join(args.out_dir, "test")
    train_path = os.path.join(args.out_dir, "train")
    stats_path = os.path.join(args.out_dir, "stats")
    metadata_path = os.path.join(args.out_dir, "metadata")
    
    for p in [test_path, train_path, stats_path, metadata_path]:
        os.makedirs(p, exist_ok=True)
        
    print(f"Generating Makani dummy dataset: {args.img_size_h}x{args.img_size_w}x{args.num_channels}")
    print(f"Path: {args.out_dir}")
    
    # Generate Stats
    c = args.num_channels
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, c, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, c, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_means.npy"), np.zeros((1, c, args.img_size_h, args.img_size_w), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, c, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, c, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, c, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, c, 1, 1), dtype=np.float64))
    
    channel_names = [f"chan_{idx}" for idx in range(c)]
    latitude = np.linspace(-90, 90, args.img_size_h, endpoint=True)[::-1]
    longitude = np.linspace(0, 360, args.img_size_w, endpoint=False)
    
    metadata = dict(
        dataset_name="testing",
        h5_path="fields",
        dims=["time", "channel", "lat", "lon"],
        dhours=(365 * 24) // args.num_samples_per_year,
        coords=dict(
            grid_type="equiangular",
            lat=latitude.tolist(),
            lon=longitude.tolist(),
            channel=channel_names,
        )
    )
    with open(os.path.join(metadata_path, "data.json"), "w") as f:
        json.dump(metadata, f)
        
    tasks = [
        (os.path.join(train_path, "2017.h5"), 2017, args.num_samples_per_year, c, args.img_size_h, args.img_size_w),
        (os.path.join(train_path, "2018.h5"), 2018, args.num_samples_per_year, c, args.img_size_h, args.img_size_w),
        (os.path.join(test_path, "2019.h5"), 2019, args.num_samples_per_year, c, args.img_size_h, args.img_size_w),
    ]
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for filepath in executor.map(create_makani_sample, tasks):
            print(f" -> Generated {filepath}")
            
    elapsed = time.time() - start_time
    total_gb = (3 * args.num_samples_per_year * c * args.img_size_h * args.img_size_w * 4) / (1024**3)
    print(f"✅ Makani dataset created in {elapsed:.1f} sec. (Total: {total_gb:.2f} GB)")

if __name__ == "__main__":
    main()
