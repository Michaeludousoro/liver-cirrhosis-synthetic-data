"""Fold-contained synthesis with the original, fixed generator settings."""
import json
from pathlib import Path

from .data_loader import (ALL_FEATURE_COLS, TARGET_COL, fit_scaler,
                          post_process_synthetic)
from .iqr_filter import filter_all
from .consensus_voting import run_consensus


def generate_fold_pools(real_training, fold, seed, output_dir=None):
    # Lazy imports keep artifact evaluation independent of TensorFlow.
    import tensorflow as tf
    from .seeding import set_global_seeds
    from .synthetic_generator import VanillaGAN, cGAN, VAE, generate_synthetic

    tf.keras.backend.clear_session()
    set_global_seeds(seed, single_thread=False)
    columns = ALL_FEATURE_COLS + [TARGET_COL]
    real = real_training[columns]
    scaler = fit_scaler(real)
    scaled = scaler.transform(real)
    models = {
        "GAN": VanillaGAN(epochs=500, print_every=100),
        "cGAN": cGAN(epochs=300, target_col_idx=columns.index(TARGET_COL), print_every=100),
        "VAE": VAE(epochs=300, print_every=100),
    }
    raw = {}
    for name, model in models.items():
        print(f"Fold {fold}: fitting {name} on {len(real)} real patients", flush=True)
        model.fit(scaled, verbose=True)
        raw[name] = generate_synthetic(model, 500, scaler, columns, post_process_synthetic)
    result = filter_all(raw, real)
    filtered = result["filtered"]
    size = min(len(filtered["GAN"]), len(filtered["cGAN"]), len(filtered["VAE"]))
    if size == 0:
        raise ValueError(f"Fold {fold}: an empty filtered pool prevents consensus; no substitution allowed")
    vae = filtered["VAE"].sample(n=size, random_state=seed).reset_index(drop=True)
    consensus, sources = run_consensus(filtered["GAN"], filtered["cGAN"], vae,
                                       tolerance=5.0, min_votes=2, verbose=True)
    if consensus.empty:
        raise ValueError(f"Fold {fold}: no consensus; no substitution allowed")
    pools = {"cgan": filtered["cGAN"], "consensus": consensus}
    if output_dir is not None:
        destination = Path(output_dir) / f"fold_{fold}"
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in raw.items():
            data.to_csv(destination / f"raw_{name}.csv", index=False)
            filtered[name].to_csv(destination / f"filtered_{name}.csv", index=False)
        consensus.to_csv(destination / "consensus.csv", index=False)
        metadata = {"fold": fold, "seed": seed,
                    "real_training_indices": real_training.index.tolist(),
                    "real_training_n": len(real), "samples_per_generator": 500,
                    "epochs": {"GAN": 500, "cGAN": 300, "VAE": 300},
                    "tolerance": 5.0, "iqr_bounds": result["bounds"],
                    "scaler_min": scaler.data_min_.tolist(),
                    "scaler_max": scaler.data_max_.tolist(),
                    "consensus_sources": sources}
        (destination / "provenance.json").write_text(json.dumps(metadata, indent=2))
    return pools
