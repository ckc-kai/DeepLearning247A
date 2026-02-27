import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate

import sys
sys.path.append(str(Path(__file__).parent.parent))

from emg2qwerty.data import EMGSessionData

def main():
    parser = argparse.ArgumentParser(description="Generate SPECTRE STFT Clusters (Pseudo-Labels)")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--k", type=int, default=500, help="Number of KMeans clusters")
    parser.add_argument("--batch_size", type=int, default=10000, help="MiniBatchKMeans batch size")
    parser.add_argument("--out", type=str, default="data/spectre_kmeans_500.pkl", help="Output pkl path")
    args = parser.parse_args()

    print(f"Initializing MiniBatchKMeans with K={args.k}...")
    # MiniBatchKMeans is much faster than standard KMeans and scales to large datasets easily
    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=args.batch_size,
        n_init="auto",
        random_state=42
    )

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="base", overrides=["model=cyro2formers_ctc"])
    
    # In this codebase, transforms are a list under `cfg.transforms.train`
    from emg2qwerty.transforms import Compose, LogSpectrogram
    
    transforms_list = []
    for t_cfg in cfg.transforms.train:
        transforms_list.append(instantiate(t_cfg))
        
    spectral_transforms = Compose(transforms_list)
    
    # Extract only the deterministic LogSpectrogram from the pipeline
    log_spec = None
    if isinstance(spectral_transforms, Compose):
        for t in spectral_transforms.transforms:
            if isinstance(t, LogSpectrogram):
                log_spec = t
                break
    else:
        log_spec = spectral_transforms
    
    assert log_spec is not None, "Could not find LogSpectrogram transform in config"
    
    print(f"Using spectrogram config: n_fft={log_spec.n_fft}, hop_length={log_spec.hop_length}")

    # Gather only the explicitly defined training sessions from the config
    # to avoid data leakage from val/test sets
    sessions = [session["session"] for session in cfg.dataset.train]
    hdf5_files = [data_dir.joinpath(f"{session}.hdf5") for session in sessions]
    
    # Filter out any files that do not exist
    hdf5_files = [path for path in hdf5_files if path.exists()]
    print(f"Found {len(hdf5_files)} training sessions.")

    # Iterate through sessions and partially fit KMeans
    for file_idx, hdf5_path in enumerate(tqdm(hdf5_files, desc="Processing Sessions")):
        with EMGSessionData(hdf5_path) as session:
            # We don't need the ground truth keystrokes, just the raw timeseries!
            timeseries = session.slice()
            
            # Extract left and right EMG
            left = torch.from_numpy(timeseries[EMGSessionData.EMG_LEFT])
            right = torch.from_numpy(timeseries[EMGSessionData.EMG_RIGHT])
            emg_tensor = torch.stack([left, right], dim=1).float()  # (T_raw, bands=2, C=16)

            # Apply STFT
            # Returns shape: (T_spec, bands=2, C=16, freq)
            spec_tensor = log_spec(emg_tensor)
            
            # SPECTRE clusters patches of the spectrogram.
            # We flatten each temporal frame into a single 1D vector.
            # Shape becomes: (T_spec, bands * C * freq)
            T_spec = spec_tensor.shape[0]
            flattened_frames = spec_tensor.reshape(T_spec, -1).numpy()
            
            # Update the KMeans model with this session's frames
            kmeans.partial_fit(flattened_frames)

    # Save the fitted model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(kmeans, f)
        
    print(f"\nSuccessfully saved K-Means model to {out_path}")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")


if __name__ == "__main__":
    main()
