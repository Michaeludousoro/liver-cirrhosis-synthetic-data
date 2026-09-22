# Manuscript revision — 16–17 September 2026

This revision starts from the September 4 manuscript in `IEEE_Synthetic Data_latex.zip`, not the older workspace manuscript. Original project sources and the supplied ZIP are preserved. The revision is an editorial and reporting correction, not a rerun of the generative experiments.

The main PDF was rebuilt and visually checked on September 17. It contains 13 pages. Citation/reference targets and LaTeX environments were checked; no reviewer placeholders remain. The review copy retains IEEE formatting pending a final venue choice. The source ZIP deliberately excludes the old supplementary PDF; the supplementary LaTeX has reporting corrections but has not undergone a full new PDF review.

## Changes made

- Replaced the long abstract with approximately 230 words and the conclusion with approximately 380 words. Counts vary slightly with treatment of mathematical notation and hyphenated terms.
- Reframed the title to describe evaluation rather than promise improved prediction or a formal theory.
- Resolved every visible red-text/reviewer placeholder in the supplied main LaTeX source, including struck-out text.
- Rewrote the introduction, model distinctions, variable definitions, consensus explanation, metric definitions, follow-up exclusion, comparator rationale, and competing-risk discussion.
- Added a selected related-work comparison, a training-only raincloud plot, and a study-overview diagram.
- Added one relevant reference from each proposed ACM journal, plus sources for Tukey fences and scikit-learn.
- Distinguished five fitted generators from the consensus-derived sixth dataset.
- Corrected FID from 18 variables to the 11 continuous variables used by `src/fid_calculator.py`; the 18-variable nearest-neighbour privacy screen is a separate calculation.
- Removed claims that non-significant results prove no effect, that McNemar tests AUC, or that a low near-duplicate rate establishes safe release.
- Corrected the reported DP-SGD sweep to approximate accounting projections rather than private training; synchronised the supplementary source on this point and the revised title.
- Retained reported numeric results. Existing figure assets have not all been regenerated; captions and source comparisons should be included in the next full results audit.

## Venue recommendation and reference placement

My editorial recommendation is to consider **Journal of Data and Information Quality (JDIQ)** first for the current contribution: assessing the agreement and disagreement between synthetic-data quality indicators and task utility. This is a fit assessment, not a prediction of acceptance. The editor's institution describes quality assessment as part of its scope: https://hpi.de/en/article/prof-naumann-reappointed-editor-in-chief-jdiq/ . Journal: https://dl.acm.org/journal/jdiq .

**ACM Transactions on Computing for Healthcare (HEALTH)** is a plausible alternative if the paper develops a stronger clinical-computing contribution and external validation. A closely relevant paper from the journal uses CTGAN augmentation and filtering in neonatal prediction. Journal: https://dl.acm.org/journal/health .

1. Michael Stenger et al. (2024), *Thinking in Categories: A Survey on Assessing the Quality for Time Series Synthesis*, Journal of Data and Information Quality. DOI: https://doi.org/10.1145/3666006 . Title, authors, venue, DOI, and abstract verified against coauthor Nathaniel Hudson's publication page: https://nathaniel-hudson.github.io/publications/ . Used to motivate structured, multidimensional synthesis evaluation, with the difference between time-series and static-tabular data stated explicitly.
2. Abdullah Mamun et al. (2026), *Use of What-if Scenarios to Help Explain Artificial Intelligence Models for Neonatal Health*, ACM Transactions on Computing for Healthcare. DOI: https://doi.org/10.1145/3814951 . Verified against the authors' laboratory publication page: https://ghasemzadeh.com/publication/2026-04-mamun-use-of-what-if-scenarios-aimen/ . Used as a clinical example of CTGAN augmentation and filtering, not as evidence that PBC performance must improve.

Both are cited in the introduction, related work, and comparison table. Their relevance is methodological rather than simply sharing the target venue. Specific volume/issue/page metadata was not inserted without primary-source verification. The discussion relies on their accessible abstracts and author publication records; full ACM article pages returned access errors. This is not a complete audit of all pre-existing references.

The JDIQ guest editor's call for a special issue on Quality of Synthetic Data listed March 3, 2026 as its submission deadline, which has passed. Do not treat that call as a currently open submission route without checking an extension: https://groups.google.com/a/aixia.it/g/aixia/c/42M3a4peWDI .

Direct ACM journal and author-guideline pages returned 403 errors during this review. The revised manuscript retains the supplied IEEE layout for review, not as a claimed ACM submission format. Confirm the selected journal's current article type, length, review template, and APC/institutional coverage before submission. No reliable journal-specific fee quote was established here.

## Proposed pitch

“This paper evaluates whether practical quality filters for synthetic clinical data translate into useful prediction on held-out real patients. A PBC case study separates distributional fidelity, empirical disclosure indicators, augmentation performance, and synthetic-only utility. The contribution is evidence about the limits of using resemblance or cross-generator agreement as a proxy for task quality, together with an evaluation framework for identifying these disagreements.”

## Scientific work still needed before submission

1. **Nested validation.** `src/statistical_analysis.py:cross_validate_scenarios` reuses synthetic pools supplied before folding. Regenerate preprocessing, generators, filtering, and selection using each training fold only. Real-only validation rows do not prevent this indirect information leakage. The current revision labels the numbers exploratory instead of silently claiming a repair.
2. **Paired inference.** McNemar concerns classification errors. Test AUC and C-index differences with appropriate paired methods and report uncertainty; do not use overlapping independent intervals as a significance test.
3. **Privacy and survival.** The approximate DP calculation is not a certified accountant or a trained DP model. The output-noise experiment needs clinical-validity and utility checks. Competing-risk prediction and independent validation remain outstanding.
4. **Reproducibility and figure audit.** Verify the exact script sequence that generated every table and figure. Confirm generator implementation settings and all legacy citations. No experiment rerun or claim of byte-identical reproducibility is made in this editorial revision.

## Build

Generate the new figures from the repository root with a working Python environment containing pandas, NumPy, SciPy, and Matplotlib:

`python revisions/2026-09-16/make_revision_figures.py`

Compile `paper/main.tex` with Tectonic or a standard LaTeX installation. The supplementary source contains corresponding wording corrections and should be rebuilt before distribution; any copied supplementary PDF from the original archive is not the revised version.
