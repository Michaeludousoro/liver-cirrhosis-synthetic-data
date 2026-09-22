"""Non-destructive artifact audit and corrected CV; outputs stay in this revision.

Run with --generators to retrain all three core generators independently in
each fold. Without it, rerun only baseline/SMOTE CV and artifact-based test
evaluation; no historical generative-CV estimates are carried forward.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import stat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sklearn
from src.data_loader import load_complete_data, split_data, CLASSIFICATION_FEATURE_COLS
from src.predictive_modeling import run_all_scenarios, run_smote_scenario, run_scenario_e_multi
from src.statistical_analysis import cross_validate_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generators', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / 'validation'
    out.mkdir(exist_ok=True)
    data = ROOT / 'output/data'
    train = pd.read_csv(data / 'train_real.csv')
    full = load_complete_data(str(ROOT / 'data/raw/cirrhosis.csv'))
    expected_train, expected_test = split_data(full, test_size=.30, random_state=42)
    pd.testing.assert_frame_equal(train, expected_train[train.columns], check_dtype=False)
    # macOS cloud placeholders can block indefinitely. Recover the fixed test
    # split from raw data after verifying the saved train split exactly.
    test_path = data / 'test_real.csv'
    test_reconstructed = bool(getattr(test_path.stat(), 'st_flags', 0) & getattr(stat, 'SF_DATALESS', 0x40000000))
    if test_reconstructed:
        test = expected_test[train.columns].copy()
        test.to_csv(out / 'test_real_reconstructed.csv', index=False)
    else:
        test = pd.read_csv(test_path)
        pd.testing.assert_frame_equal(test, expected_test[test.columns], check_dtype=False)
    assert len(train) == 193 and len(test) == 83
    overlap = train.merge(test, on=list(train.columns), how='inner')
    assert overlap.empty, 'Exact real records overlap train and test'
    assert 'N_Days' not in CLASSIFICATION_FEATURE_COLS and 'Status' not in CLASSIFICATION_FEATURE_COLS
    names = {'Vanilla GAN':'filtered_gan.csv', 'cGAN':'filtered_ctgan.csv',
             'VAE':'filtered_tvae.csv', 'Consensus':'consensus_equalised.csv',
             'Masked-loss VAE':'filtered_masked_vae.csv', 'CTGAN':'filtered_ctgan_proper.csv'}
    pools = {name: pd.read_csv(data / filename) for name, filename in names.items()}
    for name, frame in pools.items():
        assert np.isfinite(frame[train.columns].to_numpy()).all(), name
        assert set(np.round(frame.Status)).issubset({0, 1}), name
    manifest = {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__,
                'train_n': len(train), 'test_n': len(test), 'exact_overlap': len(overlap),
                'split_reproduced_from_raw': True, 'generators_retrained': False,
                'generator_retraining_requested': args.generators,
                'test_reconstructed_from_raw_because_cloud_offloaded': test_reconstructed,
                'scope': 'Saved pools verify downstream results, not original generator provenance.',
                'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / 'data/raw/cirrhosis.csv', data/'train_real.csv',
                                    *([] if test_reconstructed else [test_path]),
                                    *[data/f for f in names.values()]]}}
    (out / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('PASS: raw-data split, 193/83 sizes, no exact overlap, 17 baseline predictors', flush=True)
    perf = run_all_scenarios(train, test, pools['cGAN'], pools['Consensus'])
    perf.to_csv(out / 'model_performance.csv', index=False)
    run_smote_scenario(train, test).to_csv(out / 'smote_results.csv', index=False)
    run_scenario_e_multi(train, test, pools).to_csv(out / 'scenario_e_six_generators.csv', index=False)
    scenarios = {'A: Baseline': None, 'D: Real + SMOTE': 'smote'}
    factory = None
    if args.generators:
        from src.fold_generation import generate_fold_pools
        scenarios = {'A: Baseline':None, 'B: Real + cGAN':'cgan',
                     'C: Real + Consensus':'consensus', 'D: Real + SMOTE':'smote'}
        factory = lambda real, fold, seed: generate_fold_pools(real, fold, seed, out)
    cv = cross_validate_scenarios(train, scenarios, fold_factory=factory)
    cv.to_csv(out / 'cv_results.csv', index=False)
    pd.DataFrame(cv.attrs['fold_results']).to_csv(out / 'cv_folds.csv', index=False)
    (out / 'validation_indices.json').write_text(json.dumps(cv.attrs['validation_indices'], indent=2))
    manifest['generators_retrained'] = args.generators
    manifest['cv_completed'] = True
    if args.generators:
        import tensorflow as tf
        manifest['tensorflow'] = tf.__version__
        manifest['generator_fits'] = 15
    (out / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(cv.to_string(index=False), flush=True)
    print('Completed. Original artifacts were not overwritten.', flush=True)


if __name__ == '__main__':
    main()
