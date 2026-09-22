# ACM reviewer revision — 17 September 2026

## Read first

This revision is an ACM single-column review manuscript, with JDIQ as the working venue. It uses the unmodified ACM `acmart` class v2.20 (16 August 2026) and ACM reference style. This is an author/reviewer draft, not a claim of journal acceptance or submission readiness. Authors, affiliations, ethics statements, journal-specific submission requirements and references still require the authors' final approval.

`paper/main.tex` produces the blue-marked reviewer copy. `paper/clean.tex` uses the same content with revision colours disabled. Blue identifies changes against the IEEE source in the user's supplied ZIP, including the previous editorial corrections. Whole changed paragraphs/tables are marked, not word-by-word edits; removed passages are not struck through. Unchanged prose stays black. ACM's red margin line numbers are template numbering, not unresolved reviewer corrections. Revised figures keep their natural colours and have blue captions.

`paper/supplementary.tex` is the revised, blue-marked supplement; `paper/supplementary-clean.tex` disables marking. The source restores Figure S2 so all sixteen supplementary figures and eight tables have consistent numbering. Supplementary historical analyses are distinguished from the new validation run. All four sources were compiled and checked. The downloadable PDFs are the two main-manuscript versions; the package includes supplementary sources for compilation.

## What changed scientifically

- Removed generator leakage from cross-validation: five independent training folds now refit all three core generators, preprocessing, IQR bounds, equalisation and consensus. No precomputed full-training pools are accepted by the CV interface.
- Completed 15 generator fits: GAN 500 epochs, cGAN 300, VAE 300 in each fold, 500 candidate records per model. Seeds 42–46; split/classifier seed 42. Validation folds contain 38 or 39 real patients only. No test patients enter fitting.
- Removed the silent cGAN substitution when consensus fails. A failure now stops the run rather than changing the method being evaluated.
- Replaced the invalid generative CV results and removed fold-SD “confidence intervals”. This is retrospective, fixed-protocol, fold-contained internal validation, not nested model selection, external validation, or proof of a causal effect. The tolerance was inherited from development work on this cohort; no independence from all earlier development decisions is claimed.
- Corrected generator dimensionality: 19 generated columns = 18 variables plus Status; classifiers use 17 baseline predictors; FID uses 11 continuous variables; proximity screening uses 18 non-status variables; consensus distances use all 19 generated columns.
- Abstract and conclusion remain within 200–250 and 350–400 words, respectively. Added relevant JDIQ and ACM HEALTH literature; clarified privacy, significance, survival, and clinical-deployment limitations.

## Corrected results

Mean five-fold AUCs (RF / GB / LR):

- Real-only: 0.8499 / 0.8358 / 0.8321.
- Real + cGAN: 0.8227 / 0.8081 / 0.8184.
- Real + consensus: 0.8377 / 0.8088 / 0.7803.
- Real + within-fold SMOTE: 0.8559 / 0.8126 / 0.8300.

Both generative augmentation strategies have lower mean AUC than baseline for all three classifiers. No statistical significance is inferred from this descriptive comparison. The former consensus LR estimate of 0.8349 is superseded by 0.7803.

The audit reproduced 33 held-out classifier/result rows exactly at saved precision (12 main, 3 SMOTE, 18 six-generator synthetic-only), plus the masked-VAE headline AUCs and RF McNemar p=0.125. All six continuous-feature FIDs reproduce at reported precision. Results were initially sensitive to newer scikit-learn versions; using the original 1.5.2 reproduced the historical values.

The archived test CSV was temporarily cloud-offloaded. The fixed 83-patient test split was reconstructed from the raw data after confirming that the corresponding 193-patient training split exactly matches the saved training artifact. Original data and result files were not overwritten. The reconstructed test file and SHA-256 fingerprints are included.

## Audit scope and remaining limits

The five-fold run retrained GAN/cGAN/VAE, not the extra masked-loss VAE or CTGAN. Those two models were evaluated from saved synthetic pools. Original full-training generator weights/provenance were not independently reconstructed. Survival models, the three-year outcome analysis, privacy sweeps, subgroup analyses and same-protocol missing-data experiments were not retrained in this revision; their historical results remain explicitly limited. No formal differential-privacy guarantee, release safety, external clinical validity, or consistently beneficial augmentation is claimed.

## Reproduce from the extracted package root

Use an isolated Python environment with the versions in `requirements-validation.txt` (TensorFlow 2.15 supports Python 3.9–3.11). The completed run used Python 3.9.6 on Apple Silicon.

```sh
python revisions/2026-09-17/test_validation.py
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=42 python -u revisions/2026-09-17/audit_and_validate.py --generators
python revisions/2026-09-17/verify_artifacts.py
```

The runner writes into this revision's `validation/` directory. Preserve that directory under another name before rerunning if you wish to retain the delivered run. Running without `--generators` only audits saved pools and reruns baseline/SMOTE CV; it must not be mistaken for the full corrected CV.

Compile the manuscript from `revisions/2026-09-17/paper/`, using a standard ACM-compatible LaTeX setup or uploading that folder to Overleaf. Select `main.tex`, `clean.tex`, `supplementary.tex`, or `supplementary-clean.tex` as required. The preparation script is a development transformation tied to the supplied baseline location; edit the final LaTeX directly for subsequent manuscript revisions.

`validation/cv_folds.csv`, `validation/validation_indices.json`, each fold's `provenance.json`, and `reproduction_checks.json` record individual results and successful isolation checks. `artifact_manifest.json` records the actual runtime and hashes. Four targeted regression tests pass. The unrelated pre-existing notebook edit was left untouched.

## Reference and format sources

- ACM class source: https://ctan.org/pkg/acmart (v2.20).
- ACM review-format guidance: https://www.acm.org/binaries/content/assets/publications/taps/latex-best_practices-06-may-2020.pdf
- JDIQ: Stenger et al., “Thinking in Categories,” 16(2), 1–32 (2024), https://doi.org/10.1145/3666006. Metadata checked against ACM's Crossref deposit.
- ACM HEALTH: Mamun et al., “Use of What-if Scenarios…,” accepted manuscript online 26 May 2026, https://doi.org/10.1145/3814951. No volume or page numbers were invented.
