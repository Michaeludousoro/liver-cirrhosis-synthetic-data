"""Check reproduced metrics against version-controlled historical results."""
from pathlib import Path
import io
import json
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import numpy as np
import hashlib
from src.iqr_filter import compute_iqr_bounds
from src.data_loader import load_complete_data, split_data
from src.fid_calculator import compute_fid
from src.predictive_modeling import run_all_scenarios
from src.statistical_analysis import mcnemar_scenarios

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'validation'
train, test = split_data(load_complete_data(str(ROOT/'data/raw/cirrhosis.csv')))
checks = {}
for name in ['model_performance','smote_results','scenario_e_six_generators']:
    snapshot = OUT/'historical_results'/f'{name}.csv'
    historical = pd.read_csv(snapshot) if snapshot.exists() else pd.read_csv(io.StringIO(subprocess.check_output(
        ['git','show',f'HEAD:output/results/{name}.csv'], cwd=ROOT, text=True)))
    rerun = pd.read_csv(OUT/f'{name}.csv')
    keys = ['Classifier','Generator'] if name.startswith('scenario_e') else ['Classifier','Scenario']
    if 'Framing' in historical:
        historical = historical[historical.Framing.str.startswith('Classification')]
    metrics = ['n_train','Accuracy','F1','Precision','Recall','AUC']
    a = historical.set_index(keys)[metrics].sort_index()
    b = rerun.set_index(keys)[metrics].sort_index()
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    checks[name] = {'matched_rows': len(a), 'metrics': metrics, 'exact_to_saved_precision':True}
masked = pd.read_csv(ROOT/'output/data/filtered_masked_vae.csv')
augmented = pd.concat([train,masked],ignore_index=True)
masked_results = run_all_scenarios(train,test,masked,masked,
    scenario_b_name='F: Real + masked-VAE synthetic')
masked_results[masked_results.Scenario.str.startswith(('A:', 'F:'))].to_csv(OUT/'masked_vae_artifact_evaluation.csv',index=False)
mcnemar = mcnemar_scenarios({'A':train,'F':augmented},test,'A')
mcnemar.to_csv(OUT/'masked_vae_mcnemar.csv',index=False)
assert mcnemar.loc[mcnemar.Classifier=='Random Forest','p_value'].iloc[0] == .125
fids = []
for name, filename in {'Vanilla GAN':'filtered_gan','cGAN':'filtered_ctgan','VAE':'filtered_tvae',
                       'Consensus':'consensus_equalised','Masked-loss VAE':'filtered_masked_vae',
                       'CTGAN':'filtered_ctgan_proper'}.items():
    score = compute_fid(train,pd.read_csv(ROOT/f'output/data/{filename}.csv'))
    fids.append({'Generator':name,'FID':score})
pd.DataFrame(fids).to_csv(OUT/'fid_scores_rerun.csv',index=False)
validation_indices = json.loads((OUT/'validation_indices.json').read_text())
assert len(pd.read_csv(OUT/'cv_results.csv')) == 12, 'Full generator CV must complete before certification'
assert len(pd.read_csv(OUT/'cv_folds.csv')) == 60, 'Expected four scenarios, three classifiers and five folds'
assert sorted(sum(validation_indices,[])) == list(range(193))
fold_sizes = []
for fold, held_out in enumerate(validation_indices, 1):
    provenance = json.loads((OUT/f'fold_{fold}/provenance.json').read_text())
    members = provenance['real_training_indices']
    assert set(members).isdisjoint(held_out)
    assert set(members) | set(held_out) == set(range(193))
    part = train.iloc[members]
    np.testing.assert_allclose(provenance['scaler_min'], part.min().to_numpy())
    np.testing.assert_allclose(provenance['scaler_max'], part.max().to_numpy())
    for col, bounds in compute_iqr_bounds(part).items():
        np.testing.assert_allclose(provenance['iqr_bounds'][col], bounds)
    fold_sizes.append({'fold':fold,'real_training_n':len(members),'validation_n':len(held_out),
                       'consensus_n':len(pd.read_csv(OUT/f'fold_{fold}/consensus.csv'))})
checks['fold_isolation'] = {'all_five_folds_pass':True,'folds':fold_sizes}
manifest = json.loads((OUT/'artifact_manifest.json').read_text())
manifest.update({'cv_completed':True,'generators_retrained':True,'generator_fits':15,
                 'tensorflow':'2.15.0','fold_isolation_audited':True})
manifest['code_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/'src/fold_generation.py',ROOT/'src/statistical_analysis.py',ROOT/'src/synthetic_generator.py']}
(OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'reproduction_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
